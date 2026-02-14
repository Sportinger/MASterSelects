// SceneGraphAdapter — converts EvaluatedNode[] → Layer[] for the existing render pipeline.
// This is the migration bridge: the rest of the engine sees Layer[], unchanged.

import type { Layer, NestedCompositionData, TimelineClip } from '../../types/index.ts';
import type { EvaluatedNode } from './types.ts';
import { useMediaStore } from '../../stores/mediaStore/index.ts';

export class SceneGraphAdapter {
  /**
   * Convert evaluated nodes into the Layer[] format expected by the existing pipeline.
   * @param evaluated - Flat list of top-level evaluated nodes
   * @param clips - Current clips array for DOM ref lookup
   */
  toLayerArray(
    evaluated: EvaluatedNode[],
    clips: TimelineClip[],
    activeCompId: string
  ): Layer[] {
    const layers: Layer[] = [];
    const clipById = new Map<string, TimelineClip>();
    for (const clip of clips) {
      clipById.set(clip.id, clip);
    }

    for (const node of evaluated) {
      const layer = this.evaluatedNodeToLayer(node, clipById, activeCompId);
      if (layer) {
        layers.push(layer);
      }
    }

    return layers;
  }

  // === Private ===

  private evaluatedNodeToLayer(
    evaluated: EvaluatedNode,
    clipById: Map<string, TimelineClip>,
    activeCompId: string
  ): Layer | null {
    const { sceneNode, resolvedTransform, resolvedEffects } = evaluated;
    const clip = clipById.get(sceneNode.clipId);

    // Base layer properties
    const layerId = `${activeCompId}_layer_${sceneNode.childTrackIndex}_${sceneNode.clipId}`;

    const baseLayer: Omit<Layer, 'source'> = {
      id: layerId,
      name: clip?.name ?? sceneNode.clipId,
      visible: true,
      opacity: sceneNode.transitionOpacityMultiplier !== undefined
        ? resolvedTransform.opacity * sceneNode.transitionOpacityMultiplier
        : resolvedTransform.opacity,
      blendMode: resolvedTransform.blendMode,
      effects: resolvedEffects,
      position: resolvedTransform.position,
      scale: resolvedTransform.scale,
      rotation: resolvedTransform.rotation,
    };

    // Add mask properties if applicable
    if (sceneNode.masks && sceneNode.masks.length > 0) {
      (baseLayer as any).maskClipId = sceneNode.clipId;
      (baseLayer as any).maskInvert = sceneNode.masks.some(m => m.inverted);
    }

    // Build source based on node type
    switch (sceneNode.type) {
      case 'composition':
        return this.buildNestedCompLayer(evaluated, baseLayer, clipById, activeCompId);
      case 'video':
        return this.buildVideoLayer(evaluated, baseLayer, clip);
      case 'image':
        return this.buildImageLayer(baseLayer, clip);
      case 'text':
        return this.buildTextLayer(baseLayer, clip);
      case 'solid':
        return this.buildTextLayer(baseLayer, clip); // Solids also use textCanvas
      default:
        return null;
    }
  }

  private buildNestedCompLayer(
    evaluated: EvaluatedNode,
    baseLayer: Omit<Layer, 'source'>,
    clipById: Map<string, TimelineClip>,
    _activeCompId: string
  ): Layer | null {
    const { sceneNode } = evaluated;
    const nestedChildren: EvaluatedNode[] = (evaluated as any).nestedEvaluatedChildren ?? [];

    // Convert nested evaluated children to Layer[]
    const nestedLayers: Layer[] = [];
    const clip = clipById.get(sceneNode.clipId);

    if (clip?.nestedClips) {
      const nestedClipById = new Map<string, TimelineClip>();
      for (const nc of clip.nestedClips) {
        nestedClipById.set(nc.id, nc);
      }

      for (const childEval of nestedChildren) {
        const childClip = nestedClipById.get(childEval.sceneNode.clipId);
        if (!childClip) continue;

        const childLayer = this.buildNestedChildLayer(childEval, childClip);
        if (childLayer) {
          nestedLayers.push(childLayer);
        }
      }
    }

    if (nestedLayers.length === 0) return null;

    // Get composition dimensions
    const compositions = useMediaStore.getState().compositions;
    const composition = compositions.find(c => c.id === sceneNode.compositionId);
    const compWidth = composition?.width ?? 1920;
    const compHeight = composition?.height ?? 1080;

    const nestedCompData: NestedCompositionData = {
      compositionId: sceneNode.compositionId ?? sceneNode.clipId,
      layers: nestedLayers,
      width: compWidth,
      height: compHeight,
      currentTime: evaluated.sourceTime,
    };

    return {
      ...baseLayer,
      source: { type: 'image', nestedComposition: nestedCompData },
    } as Layer;
  }

  private buildNestedChildLayer(
    evaluated: EvaluatedNode,
    clip: TimelineClip
  ): Layer | null {
    const { resolvedTransform, resolvedEffects, sceneNode } = evaluated;

    if (clip.isLoading) return null;

    const baseLayer = {
      id: `nested-layer-${sceneNode.clipId}`,
      name: clip.name,
      visible: true,
      opacity: resolvedTransform.opacity,
      blendMode: resolvedTransform.blendMode,
      effects: resolvedEffects,
      position: resolvedTransform.position,
      scale: resolvedTransform.scale,
      rotation: resolvedTransform.rotation,
    };

    // Add mask properties
    if (sceneNode.masks && sceneNode.masks.length > 0) {
      (baseLayer as any).maskClipId = sceneNode.clipId;
      (baseLayer as any).maskInvert = sceneNode.masks.some(m => m.inverted);
    }

    // Composition child → recursive nested comp (multi-level nesting)
    if (clip.isComposition && clip.nestedClips) {
      const nestedChildren: EvaluatedNode[] = (evaluated as any).nestedEvaluatedChildren ?? [];
      const nestedClipById = new Map<string, TimelineClip>();
      for (const nc of clip.nestedClips) {
        nestedClipById.set(nc.id, nc);
      }

      const nestedLayers: Layer[] = [];
      for (const childEval of nestedChildren) {
        const childClip = nestedClipById.get(childEval.sceneNode.clipId);
        if (!childClip) continue;
        const childLayer = this.buildNestedChildLayer(childEval, childClip);
        if (childLayer) nestedLayers.push(childLayer);
      }

      if (nestedLayers.length === 0) return null;

      const compositions = useMediaStore.getState().compositions;
      const composition = compositions.find(c => c.id === sceneNode.compositionId);
      const compWidth = composition?.width ?? 1920;
      const compHeight = composition?.height ?? 1080;

      const nestedCompData: NestedCompositionData = {
        compositionId: sceneNode.compositionId ?? sceneNode.clipId,
        layers: nestedLayers,
        width: compWidth,
        height: compHeight,
        currentTime: evaluated.sourceTime,
      };

      return {
        ...baseLayer,
        source: { type: 'image', nestedComposition: nestedCompData },
      } as Layer;
    }

    if (clip.source?.nativeDecoder) {
      return {
        ...baseLayer,
        source: { type: 'video' as const, nativeDecoder: clip.source.nativeDecoder },
      } as Layer;
    }

    if (clip.source?.videoElement) {
      return {
        ...baseLayer,
        source: {
          type: 'video' as const,
          videoElement: clip.source.videoElement,
          webCodecsPlayer: clip.source.webCodecsPlayer,
        },
      } as Layer;
    }

    if (clip.source?.imageElement) {
      return {
        ...baseLayer,
        source: { type: 'image' as const, imageElement: clip.source.imageElement },
      } as Layer;
    }

    if (clip.source?.textCanvas) {
      return {
        ...baseLayer,
        source: { type: 'text' as const, textCanvas: clip.source.textCanvas },
      } as Layer;
    }

    return null;
  }

  private buildVideoLayer(
    _evaluated: EvaluatedNode,
    baseLayer: Omit<Layer, 'source'>,
    clip?: TimelineClip
  ): Layer | null {
    if (!clip?.source) return null;

    // Native decoder
    if (clip.source.nativeDecoder) {
      return {
        ...baseLayer,
        source: { type: 'video', nativeDecoder: clip.source.nativeDecoder },
      } as Layer;
    }

    // Video element (+ optional WebCodecs)
    if (clip.source.videoElement) {
      return {
        ...baseLayer,
        source: {
          type: 'video',
          videoElement: clip.source.videoElement,
          webCodecsPlayer: clip.source.webCodecsPlayer,
        },
      } as Layer;
    }

    return null;
  }

  private buildImageLayer(
    baseLayer: Omit<Layer, 'source'>,
    clip?: TimelineClip
  ): Layer | null {
    if (!clip?.source?.imageElement) return null;

    return {
      ...baseLayer,
      source: { type: 'image', imageElement: clip.source.imageElement },
    } as Layer;
  }

  private buildTextLayer(
    baseLayer: Omit<Layer, 'source'>,
    clip?: TimelineClip
  ): Layer | null {
    if (!clip?.source?.textCanvas) return null;

    return {
      ...baseLayer,
      source: { type: 'text', textCanvas: clip.source.textCanvas },
    } as Layer;
  }
}
