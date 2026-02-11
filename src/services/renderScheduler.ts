// RenderScheduler - Unified render loop for all independent render targets
// Replaces PreviewRenderManager with store-based target resolution
// Handles composition, layer, and slot sources independently from the main render loop

import { Logger } from './logger';
import type { Layer } from '../types';
import type { RenderSource } from '../types/renderTarget';

const log = Logger.create('RenderScheduler');
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { useRenderTargetStore } from '../stores/renderTargetStore';
import { compositionRenderer } from './compositionRenderer';
import { engine } from '../engine/WebGPUEngine';

interface NestedCompInfo {
  clipId: string;
  clipStartTime: number;
  clipDuration: number;
  clipInPoint: number;
  clipOutPoint: number;
}

class RenderSchedulerService {
  private registeredTargets: Set<string> = new Set();
  private preparedCompositions: Set<string> = new Set();
  // Track compositions currently being prepared (to avoid duplicate prepareComposition calls)
  private preparingCompositions: Set<string> = new Set();
  private rafId: number | null = null;
  private isRunning = false;
  private lastFrameTime = 0;

  // Cache nested composition info to avoid recalculating every frame
  private nestedCompCache: Map<string, NestedCompInfo | null> = new Map();
  private nestedCompCacheTime = 0;
  private readonly CACHE_INVALIDATION_MS = 100;

  // Reuse the main loop's pre-built layers for the active composition
  // Avoids re-seeking video elements and bypasses compositionRenderer entirely
  private activeCompLayers: Layer[] | null = null;

  /**
   * Register a render target for independent rendering
   */
  register(targetId: string): void {
    log.debug(`Registering target: ${targetId}`);
    this.registeredTargets.add(targetId);

    // Resolve source to compositionId and prepare it
    const target = useRenderTargetStore.getState().targets.get(targetId);
    if (target) {
      const compId = useRenderTargetStore.getState().resolveSourceToCompId(target.source);
      if (compId && !this.preparedCompositions.has(compId)) {
        this.preparedCompositions.add(compId);
        compositionRenderer.prepareComposition(compId).then((ready) => {
          log.debug(`Composition ${compId} ready: ${ready}`);
        });
      }
    }

    this.startLoop();
  }

  /**
   * Unregister a render target
   */
  unregister(targetId: string): void {
    log.debug(`Unregistering target: ${targetId}`);
    this.registeredTargets.delete(targetId);

    if (this.registeredTargets.size === 0) {
      this.stopLoop();
    }
  }

  /**
   * Notify the scheduler that a target's source changed
   * Re-prepares the new composition if needed
   */
  updateTargetSource(targetId: string): void {
    if (!this.registeredTargets.has(targetId)) return;

    const target = useRenderTargetStore.getState().targets.get(targetId);
    if (target) {
      const compId = useRenderTargetStore.getState().resolveSourceToCompId(target.source);
      if (compId && !this.preparedCompositions.has(compId)) {
        this.preparedCompositions.add(compId);
        compositionRenderer.prepareComposition(compId).then((ready) => {
          log.debug(`Composition ${compId} ready after source change: ${ready}`);
        });
      }
    }
  }

  /**
   * Check if a composition is nested in the active timeline and return its info
   */
  private getNestedCompInfo(compositionId: string): NestedCompInfo | null {
    const now = Date.now();

    if (now - this.nestedCompCacheTime < this.CACHE_INVALIDATION_MS) {
      const cached = this.nestedCompCache.get(compositionId);
      if (cached !== undefined) return cached;
    }

    const mainClips = useTimelineStore.getState().clips;
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
    const activeCompId = useMediaStore.getState().activeCompositionId;

    // Case 1: This composition is nested in active timeline (child preview while parent is active)
    const nestedInfo = this.getNestedCompInfo(compositionId);
    if (nestedInfo) {
      const clipStart = nestedInfo.clipStartTime;
      const clipEnd = clipStart + nestedInfo.clipDuration;

      if (mainPlayhead >= clipStart && mainPlayhead < clipEnd) {
        const relativeTime = mainPlayhead - clipStart;
        const compositionTime = relativeTime + nestedInfo.clipInPoint;
        return { time: compositionTime, syncSource: 'nested' };
      }

      if (mainPlayhead < clipStart) {
        return { time: nestedInfo.clipInPoint, syncSource: 'nested' };
      }

      if (mainPlayhead >= clipEnd) {
        return { time: nestedInfo.clipOutPoint, syncSource: 'nested' };
      }
    }

    // Case 2: Active composition is nested in THIS composition (parent preview while child is active)
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
          const parentTime = clipStart + (mainPlayhead - inPoint);
          return { time: parentTime, syncSource: 'reverse-nested' };
        }
      }
    }

    // Not nested - use composition's own stored playhead
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

    log.info('Starting render scheduler loop');
    this.isRunning = true;
    this.lastFrameTime = performance.now();

    const renderLoop = () => {
      if (!this.isRunning) return;

      const now = performance.now();
      const deltaTime = now - this.lastFrameTime;
      this.lastFrameTime = now;

      // Throttle to ~60fps (16.67ms) with slight buffer
      const shouldRender = deltaTime >= 14;

      // Skip during export to prevent video element conflicts
      if (shouldRender && !engine.getIsExporting()) {
        this.renderAllTargets();
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

    log.info('Stopping render scheduler loop');
    this.isRunning = false;

    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  /**
   * Render all registered independent targets
   */
  private renderAllTargets(): void {
    const now = Date.now();
    if (now - this.nestedCompCacheTime >= this.CACHE_INVALIDATION_MS) {
      this.nestedCompCache.clear();
    }

    const activeCompId = useMediaStore.getState().activeCompositionId;
    const store = useRenderTargetStore.getState();

    // Per-frame evaluation cache: evaluate each composition only once
    const evalCache = new Map<string, Layer[]>();

    for (const targetId of this.registeredTargets) {
      const target = store.targets.get(targetId);
      if (!target || !target.enabled) continue;

      // Resolve source to compositionId
      const compId = store.resolveSourceToCompId(target.source);

      if (!compId) {
        // Empty slot or unresolvable source — render black
        engine.renderToPreviewCanvas(targetId, []);
        continue;
      }

      // Skip if active comp — main loop handles it
      // Exception: layer-filtered sources need independent rendering even for active comp
      const needsIndependentRender = target.source.type === 'layer' || target.source.type === 'layer-index';
      if (compId === activeCompId && !needsIndependentRender) continue;

      // For active comp with layer filtering: reuse pre-built layers from main loop
      // This avoids re-seeking video elements and re-evaluating the same composition
      if (compId === activeCompId && needsIndependentRender && this.activeCompLayers) {
        let filtered: Layer[];
        if (target.source.type === 'layer') {
          const layerIds = target.source.layerIds;
          filtered = this.activeCompLayers.filter(l => layerIds.includes(l.id));
        } else if (target.source.type === 'layer-index') {
          const idx = target.source.layerIndex;
          filtered = idx < this.activeCompLayers.length ? [this.activeCompLayers[idx]] : [];
        } else {
          filtered = this.activeCompLayers;
        }
        engine.renderToPreviewCanvas(targetId, filtered);
        continue;
      }

      // Optimization: copy pre-rendered nested comp texture instead of re-rendering
      const nestedInfo = this.getNestedCompInfo(compId);
      if (nestedInfo) {
        const mainPlayhead = useTimelineStore.getState().playheadPosition;
        const clipStart = nestedInfo.clipStartTime;
        const clipEnd = clipStart + nestedInfo.clipDuration;

        if (mainPlayhead >= clipStart && mainPlayhead < clipEnd) {
          if (engine.copyNestedCompTextureToPreview(targetId, compId)) {
            continue; // Reused pre-rendered texture
          }
          // Fall through to independent rendering
        }
      }

      // If composition isn't ready, trigger (re-)preparation and skip this frame
      if (!compositionRenderer.isReady(compId)) {
        if (!this.preparingCompositions.has(compId)) {
          this.preparingCompositions.add(compId);
          compositionRenderer.prepareComposition(compId).then((ready) => {
            this.preparingCompositions.delete(compId);
            log.debug(`Composition ${compId} auto-prepared in render loop: ${ready}`);
          });
        }
        continue;
      }

      // Get or evaluate layers (cached per composition per frame)
      let evalLayers: Layer[];
      if (evalCache.has(compId)) {
        evalLayers = evalCache.get(compId)!;
      } else {
        const { time: playheadTime } = this.calculatePlayheadTime(compId);
        evalLayers = compositionRenderer.evaluateAtTime(compId, playheadTime) as Layer[];
        evalCache.set(compId, evalLayers);
      }

      // Layer filtering: if source targets specific layers, filter
      if (target.source.type === 'layer') {
        const layerIds = target.source.layerIds;
        evalLayers = evalLayers.filter(l => layerIds.includes(l.id));
      } else if (target.source.type === 'layer-index') {
        const idx = target.source.layerIndex;
        evalLayers = idx < evalLayers.length ? [evalLayers[idx]] : [];
      }

      // Render to the target canvas (empty = black)
      engine.renderToPreviewCanvas(targetId, evalLayers);
    }
  }

  /**
   * Force re-render all targets (e.g., after composition changes)
   */
  forceRender(): void {
    this.nestedCompCache.clear();
    this.renderAllTargets();
  }

  /**
   * Invalidate the nested composition cache (call when clips change)
   */
  invalidateNestedCache(): void {
    this.nestedCompCache.clear();
    this.nestedCompCacheTime = 0;
  }

  /**
   * Set the active composition's pre-built layers from the main render loop.
   * Called by useEngine after buildLayersFromStore() — avoids re-seeking videos
   * and re-evaluating the same composition in the renderScheduler.
   */
  setActiveCompLayers(layers: Layer[]): void {
    this.activeCompLayers = layers;
  }

  /**
   * Get debug info about registered targets
   */
  getDebugInfo(): { targetId: string; source: RenderSource; compositionId: string | null }[] {
    const store = useRenderTargetStore.getState();
    return Array.from(this.registeredTargets).map(targetId => {
      const target = store.targets.get(targetId);
      return {
        targetId,
        source: target?.source ?? { type: 'activeComp' as const },
        compositionId: target ? store.resolveSourceToCompId(target.source) : null,
      };
    });
  }
}

// Singleton instance
export const renderScheduler = new RenderSchedulerService();
