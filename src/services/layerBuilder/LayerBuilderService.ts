// LayerBuilderService - Main orchestrator for layer building
// Delegates video sync to VideoSyncManager and audio sync to AudioTrackSyncManager

import type { Layer } from '../../types';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { VideoSyncManager } from './VideoSyncManager';
import { AudioTrackSyncManager } from './AudioTrackSyncManager';
import { SceneGraphBuilder } from '../../engine/sceneGraph/SceneGraphBuilder';
import { SceneGraphEvaluator } from '../../engine/sceneGraph/SceneGraphEvaluator';
import { SceneGraphAdapter } from '../../engine/sceneGraph/SceneGraphAdapter';

/**
 * LayerBuilderService - Builds render layers from timeline state
 * Uses Scene Graph pipeline: SceneGraph → EvaluatedNode[] → Layer[]
 */
export class LayerBuilderService {
  private videoSyncManager = new VideoSyncManager();
  private audioTrackSyncManager = new AudioTrackSyncManager();

  // Scene Graph pipeline
  private sceneGraphBuilder = new SceneGraphBuilder();
  private sceneGraphEvaluator = new SceneGraphEvaluator();
  private sceneGraphAdapter = new SceneGraphAdapter();

  /**
   * Invalidate scene graph cache (forces rebuild next frame)
   */
  invalidateCache(): void {
    this.sceneGraphBuilder.invalidate();
  }

  /**
   * Build layers for the current frame
   * Main entry point - called from render loop
   */
  buildLayersFromStore(): Layer[] {
    // Scene Graph path — always active
    return this.buildLayersViaSceneGraph();
  }

  /**
   * Build layers using the Scene Graph pipeline (Phase 1).
   * Produces identical Layer[] output, but via SceneGraph → EvaluatedNode[] → Layer[].
   */
  private buildLayersViaSceneGraph(): Layer[] {
    const mediaState = useMediaStore.getState();
    const activeCompId = mediaState.activeCompositionId;
    if (!activeCompId) {
      return this.mergeBackgroundLayers([], useTimelineStore.getState().playheadPosition);
    }

    const timelineState = useTimelineStore.getState();
    const playheadPosition = timelineState.playheadPosition;

    const graph = this.sceneGraphBuilder.build();
    const evaluated = this.sceneGraphEvaluator.evaluate(graph, playheadPosition);
    const primaryLayers = this.sceneGraphAdapter.toLayerArray(evaluated, timelineState.clips, activeCompId);

    return this.mergeBackgroundLayers(primaryLayers, playheadPosition);
  }

  /**
   * Merge primary (editor) layers with background composition layers.
   * Render order: D (bottom) → C → B → A (top)
   * The primary composition's layers go at the position of its layer slot.
   */
  private mergeBackgroundLayers(primaryLayers: Layer[], playheadPosition: number): Layer[] {
    const { activeLayerSlots, activeCompositionId } = useMediaStore.getState();
    const slotEntries = Object.entries(activeLayerSlots);

    // No active layer slots → return primary layers as-is (backwards compatible)
    if (slotEntries.length === 0) {
      return primaryLayers;
    }

    // Find which layer the primary (editor) composition is on
    let primaryLayerIndex = -1;
    for (const [key, compId] of slotEntries) {
      if (compId === activeCompositionId) {
        primaryLayerIndex = Number(key);
        break;
      }
    }

    // Collect all layer indices, sorted A=0 (top) → D=3 (bottom)
    // layers[0] is rendered last (on top) by the compositor's reverse iteration
    const layerIndices = slotEntries
      .map(([key]) => Number(key))
      .sort((a, b) => a - b); // Ascending: A=0 first (top) → D=3 last (bottom)

    const merged: Layer[] = [];

    const { layerOpacities } = useMediaStore.getState();

    for (const layerIndex of layerIndices) {
      if (layerIndex === primaryLayerIndex) {
        // Insert primary layers at this position, applying layer opacity
        const layerOpacity = layerOpacities[layerIndex] ?? 1;
        // Filter out undefined entries from sparse arrays (buildLayers uses layers[trackIndex]=...)
        const actualPrimaryLayers = primaryLayers.filter((l): l is Layer => l != null);
        if (layerOpacity < 1 && actualPrimaryLayers.length > 0) {
          // Apply per-clip opacity multiplication — simpler and works with all decoder types
          // (nativeDecoder, WebCodecs, HTMLVideo, etc.) without needing NestedCompRenderer
          for (const layer of actualPrimaryLayers) {
            merged.push({ ...layer, opacity: layer.opacity * layerOpacity });
          }
        } else {
          merged.push(...actualPrimaryLayers);
        }
      } else {
        // Build background layer from LayerPlaybackManager
        const bgLayer = layerPlaybackManager.buildLayersForLayer(layerIndex, playheadPosition);
        if (bgLayer) {
          merged.push(bgLayer);
        }
      }
    }

    // If primary comp is not in any slot, add its layers on top
    if (primaryLayerIndex === -1 && primaryLayers.length > 0) {
      merged.push(...primaryLayers.filter((l): l is Layer => l != null));
    }

    return merged;
  }

  // ==================== VIDEO & AUDIO SYNC (delegated) ====================

  /**
   * Prepare continuous playback for contiguous same-source clips.
   * Swaps video elements so clips inherit the playing video from the previous clip.
   * Call BEFORE buildLayersFromStore().
   */
  prepareContinuousPlayback(): void {
    this.videoSyncManager.prepareContinuousPlayback();
  }

  /**
   * Finalize prerolled clips before render — pauses prerolled videos
   * and seeks to correct position so first render frame is correct.
   * Call BEFORE engine.render().
   */
  finalizePrerolls(): void {
    this.videoSyncManager.finalizePrerolls();
  }

  /**
   * Sync video elements to current playhead
   */
  syncVideoElements(): void {
    this.videoSyncManager.syncVideoElements();
  }

  /**
   * Sync audio elements to current playhead
   */
  syncAudioElements(): void {
    this.audioTrackSyncManager.syncAudioElements();
  }
}
