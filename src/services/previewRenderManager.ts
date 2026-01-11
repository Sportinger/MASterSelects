// PreviewRenderManager - Centralized render loop for all independent preview canvases
// Single RAF loop handles all previews efficiently with proper nested composition sync

import type { Layer } from '../types';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { compositionRenderer } from './compositionRenderer';
import { engine } from '../engine/WebGPUEngine';

interface RegisteredPreview {
  panelId: string;
  compositionId: string;
  isReady: boolean;
  lastRenderTime: number;
}

interface NestedCompInfo {
  clipId: string;
  clipStartTime: number;
  clipDuration: number;
  clipInPoint: number;
  clipOutPoint: number;
}

class PreviewRenderManagerService {
  private registeredPreviews: Map<string, RegisteredPreview> = new Map();
  private rafId: number | null = null;
  private isRunning = false;
  private lastFrameTime = 0;

  // Cache nested composition info to avoid recalculating every frame
  private nestedCompCache: Map<string, NestedCompInfo | null> = new Map();
  private nestedCompCacheTime = 0;
  private readonly CACHE_INVALIDATION_MS = 100; // Refresh cache every 100ms

  /**
   * Register a preview panel for rendering
   */
  register(panelId: string, compositionId: string): void {
    console.log(`[PreviewRenderManager] Registering preview: ${panelId} for composition: ${compositionId}`);

    this.registeredPreviews.set(panelId, {
      panelId,
      compositionId,
      isReady: false,
      lastRenderTime: 0,
    });

    // Prepare the composition
    compositionRenderer.prepareComposition(compositionId).then((ready) => {
      const preview = this.registeredPreviews.get(panelId);
      if (preview) {
        preview.isReady = ready;
        console.log(`[PreviewRenderManager] Composition ${compositionId} ready: ${ready}`);
      }
    });

    // Start the loop if not running
    this.startLoop();
  }

  /**
   * Unregister a preview panel
   */
  unregister(panelId: string): void {
    console.log(`[PreviewRenderManager] Unregistering preview: ${panelId}`);
    this.registeredPreviews.delete(panelId);

    // Stop loop if no more previews
    if (this.registeredPreviews.size === 0) {
      this.stopLoop();
    }
  }

  /**
   * Update the composition for a preview panel
   */
  updateComposition(panelId: string, compositionId: string): void {
    const existing = this.registeredPreviews.get(panelId);
    if (existing && existing.compositionId !== compositionId) {
      this.unregister(panelId);
      this.register(panelId, compositionId);
    }
  }

  /**
   * Check if a composition is nested in the active timeline and return its info
   */
  private getNestedCompInfo(compositionId: string): NestedCompInfo | null {
    const now = Date.now();

    // Use cached value if still valid
    if (now - this.nestedCompCacheTime < this.CACHE_INVALIDATION_MS) {
      const cached = this.nestedCompCache.get(compositionId);
      if (cached !== undefined) {
        return cached;
      }
    }

    // Refresh cache
    const mainClips = useTimelineStore.getState().clips;

    // Find if this composition is nested as a clip in the active timeline
    const nestedClip = mainClips.find(c => c.isComposition && c.compositionId === compositionId);

    let info: NestedCompInfo | null = null;
    if (nestedClip) {
      info = {
        clipId: nestedClip.id,
        clipStartTime: nestedClip.startTime,
        clipDuration: nestedClip.duration,
        clipInPoint: nestedClip.inPoint || 0,
        clipOutPoint: nestedClip.outPoint || nestedClip.duration,
      };
    }

    this.nestedCompCache.set(compositionId, info);
    this.nestedCompCacheTime = now;

    return info;
  }

  /**
   * Calculate the playhead time for a composition
   * If nested in active timeline and main playhead is within the clip, sync to it
   * If active composition is nested in this composition, sync from child to parent
   * Otherwise, use the composition's own stored playhead
   */
  private calculatePlayheadTime(compositionId: string): { time: number; syncSource: 'nested' | 'reverse-nested' | 'stored' | 'default' } {
    const mainPlayhead = useTimelineStore.getState().playheadPosition;
    const isMainPlaying = useTimelineStore.getState().isPlaying;
    const activeCompId = useMediaStore.getState().activeCompositionId;

    // Case 1: Check if this composition is nested in active timeline (child preview while parent is active)
    const nestedInfo = this.getNestedCompInfo(compositionId);

    if (nestedInfo) {
      const clipStart = nestedInfo.clipStartTime;
      const clipEnd = clipStart + nestedInfo.clipDuration;

      // If main playhead is within this nested clip's range, sync to it
      if (mainPlayhead >= clipStart && mainPlayhead < clipEnd) {
        // Calculate internal composition time:
        // (main playhead position relative to clip start) + clip's in-point
        const relativeTime = mainPlayhead - clipStart;
        const compositionTime = relativeTime + nestedInfo.clipInPoint;

        return { time: compositionTime, syncSource: 'nested' };
      }

      // If playhead is before the nested clip, show at in-point
      // (whether playing or scrubbing - always stay synced to parent)
      if (mainPlayhead < clipStart) {
        return { time: nestedInfo.clipInPoint, syncSource: 'nested' };
      }

      // If playhead is after the nested clip, show at out-point
      if (mainPlayhead >= clipEnd) {
        return { time: nestedInfo.clipOutPoint, syncSource: 'nested' };
      }
    }

    // Case 2: Check if the ACTIVE composition is nested in THIS composition (parent preview while child is active)
    // This enables parent preview to update when playing the child composition
    if (activeCompId && activeCompId !== compositionId) {
      const composition = useMediaStore.getState().compositions.find(c => c.id === compositionId);
      if (composition?.timelineData?.clips) {
        const childClip = composition.timelineData.clips.find(
          (c: { isComposition?: boolean; compositionId?: string }) =>
            c.isComposition && c.compositionId === activeCompId
        );
        if (childClip) {
          const clipStart = childClip.startTime;
          const inPoint = childClip.inPoint || 0;
          // Calculate parent's playhead from child's playhead
          // parentTime = clipStart + (childPlayhead - inPoint)
          const parentTime = clipStart + (mainPlayhead - inPoint);
          return { time: parentTime, syncSource: 'reverse-nested' };
        }
      }
    }

    // Not nested or not in range - use composition's own stored playhead
    const composition = useMediaStore.getState().compositions.find(c => c.id === compositionId);
    if (composition?.timelineData?.playheadPosition !== undefined) {
      return { time: composition.timelineData.playheadPosition, syncSource: 'stored' };
    }

    return { time: 0, syncSource: 'default' };
  }

  /**
   * Start the unified render loop
   */
  private startLoop(): void {
    if (this.isRunning) return;

    console.log('[PreviewRenderManager] Starting unified render loop');
    this.isRunning = true;
    this.lastFrameTime = performance.now();

    const renderLoop = () => {
      if (!this.isRunning) return;

      const now = performance.now();
      const deltaTime = now - this.lastFrameTime;
      this.lastFrameTime = now;

      // Throttle to ~60fps max (16.67ms)
      // But allow catch-up if we're behind
      const shouldRender = deltaTime >= 14; // Slight buffer under 16.67ms

      if (shouldRender) {
        this.renderAllPreviews();
      }

      this.rafId = requestAnimationFrame(renderLoop);
    };

    this.rafId = requestAnimationFrame(renderLoop);
  }

  /**
   * Stop the render loop
   */
  private stopLoop(): void {
    if (!this.isRunning) return;

    console.log('[PreviewRenderManager] Stopping unified render loop');
    this.isRunning = false;

    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  /**
   * Render all registered preview canvases in one pass
   */
  private renderAllPreviews(): void {
    // Clear the nested comp cache periodically
    const now = Date.now();
    if (now - this.nestedCompCacheTime >= this.CACHE_INVALIDATION_MS) {
      this.nestedCompCache.clear();
    }

    const mainPlayhead = useTimelineStore.getState().playheadPosition;
    const activeCompId = useMediaStore.getState().activeCompositionId;

    for (const preview of this.registeredPreviews.values()) {
      if (!preview.isReady) continue;

      // OPTIMIZATION 1: If this preview shows the ACTIVE composition,
      // skip independent rendering - main render loop handles it automatically
      // (Main loop checks independentCanvasCompositions to render to matching canvases)
      if (preview.compositionId === activeCompId) {
        preview.lastRenderTime = now;
        continue; // Skip independent rendering - main loop handles it
      }

      // OPTIMIZATION 2: If this composition is nested and currently being rendered by main loop,
      // copy the pre-rendered nested texture instead of re-rendering
      const nestedInfo = this.getNestedCompInfo(preview.compositionId);
      if (nestedInfo) {
        const clipStart = nestedInfo.clipStartTime;
        const clipEnd = clipStart + nestedInfo.clipDuration;

        // If main playhead is within this nested clip's range, try to reuse the pre-rendered texture
        // This avoids video frame conflicts by copying the already-rendered texture from main loop
        if (mainPlayhead >= clipStart && mainPlayhead < clipEnd) {
          // Try to copy the pre-rendered nested comp texture to this preview
          if (engine.copyNestedCompTextureToPreview(preview.panelId, preview.compositionId)) {
            preview.lastRenderTime = now;
            continue; // Success - skip independent rendering
          }
          // Copy failed (texture not ready yet) - fall through to independent rendering
        }
      }

      // Calculate playhead time for this composition
      const { time: playheadTime, syncSource } = this.calculatePlayheadTime(preview.compositionId);

      // OPTIMIZATION 3: If this preview contains the active comp as a nested clip (reverse-nested),
      // try to use the cached active comp texture instead of re-rendering (avoids video conflicts)
      const isMainPlaying = useTimelineStore.getState().isPlaying;
      if (syncSource === 'reverse-nested' && isMainPlaying && activeCompId) {
        // Try to copy the cached active comp texture
        if (engine.copyNestedCompTextureToPreview(preview.panelId, activeCompId)) {
          preview.lastRenderTime = now;
          continue;
        }
        // If no cached texture, skip to avoid video conflicts
        continue;
      }

      // Otherwise: Render this composition independently (different comp or different time)
      // Evaluate the composition at this time
      const evalLayers = compositionRenderer.evaluateAtTime(preview.compositionId, playheadTime);

      // Render to the preview canvas
      if (evalLayers.length > 0) {
        engine.renderToPreviewCanvas(preview.panelId, evalLayers as Layer[]);
      }

      preview.lastRenderTime = now;
    }
  }

  /**
   * Force re-render all previews (e.g., after composition changes)
   */
  forceRender(): void {
    this.nestedCompCache.clear();
    this.renderAllPreviews();
  }

  /**
   * Get debug info about registered previews
   */
  getDebugInfo(): { panelId: string; compositionId: string; isReady: boolean }[] {
    return Array.from(this.registeredPreviews.values()).map(p => ({
      panelId: p.panelId,
      compositionId: p.compositionId,
      isReady: p.isReady,
    }));
  }

  /**
   * Invalidate the nested composition cache (call when clips change)
   */
  invalidateNestedCache(): void {
    this.nestedCompCache.clear();
    this.nestedCompCacheTime = 0;
  }
}

// Singleton instance
export const previewRenderManager = new PreviewRenderManagerService();
