// Keyframe-related actions slice

import type { KeyframeActions, SliceCreator, Keyframe, AnimatableProperty, ClipTransform } from './types';
import { DEFAULT_TRANSFORM, PROPERTY_ROW_HEIGHT, MIN_CURVE_EDITOR_HEIGHT, MAX_CURVE_EDITOR_HEIGHT } from './constants';
import {
  getInterpolatedClipTransform,
  getKeyframeAtTime,
  hasKeyframesForProperty,
  interpolateKeyframes
} from '../../utils/keyframeInterpolation';
import { composeTransforms } from '../../utils/transformComposition';
import { calculateSourceTime, getSpeedAtTime, calculateTimelineDuration } from '../../utils/speedIntegration';

export const createKeyframeSlice: SliceCreator<KeyframeActions> = (set, get) => ({
  addKeyframe: (clipId, property, value, time, easing = 'linear') => {
    const { clips, playheadPosition, clipKeyframes, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    // Calculate time relative to clip start
    const clipLocalTime = time ?? (playheadPosition - clip.startTime);

    // Clamp to clip duration
    const clampedTime = Math.max(0, Math.min(clipLocalTime, clip.duration));

    // Get existing keyframes for this clip
    const existingKeyframes = clipKeyframes.get(clipId) || [];

    // Check if keyframe already exists at this time for this property
    const existingAtTime = getKeyframeAtTime(existingKeyframes, property, clampedTime);

    let newKeyframes: Keyframe[];

    if (existingAtTime) {
      // Update existing keyframe
      newKeyframes = existingKeyframes.map(k =>
        k.id === existingAtTime.id ? { ...k, value, easing } : k
      );
    } else {
      // Create new keyframe
      const newKeyframe: Keyframe = {
        id: `kf_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
        clipId,
        time: clampedTime,
        property,
        value,
        easing,
      };
      newKeyframes = [...existingKeyframes, newKeyframe].sort((a, b) => a.time - b.time);
    }

    // Update state
    const newMap = new Map(clipKeyframes);
    newMap.set(clipId, newKeyframes);
    set({ clipKeyframes: newMap });

    // Invalidate cache since animation changed
    invalidateCache();
  },

  removeKeyframe: (keyframeId) => {
    const { clipKeyframes, invalidateCache, selectedKeyframeIds } = get();
    const newMap = new Map<string, Keyframe[]>();

    clipKeyframes.forEach((keyframes, clipId) => {
      const filtered = keyframes.filter(k => k.id !== keyframeId);
      if (filtered.length > 0) {
        newMap.set(clipId, filtered);
      }
    });

    // Remove from selection
    const newSelection = new Set(selectedKeyframeIds);
    newSelection.delete(keyframeId);

    set({ clipKeyframes: newMap, selectedKeyframeIds: newSelection });
    invalidateCache();
  },

  updateKeyframe: (keyframeId, updates) => {
    const { clipKeyframes, invalidateCache } = get();
    const newMap = new Map<string, Keyframe[]>();

    clipKeyframes.forEach((keyframes, clipId) => {
      newMap.set(clipId, keyframes.map(k =>
        k.id === keyframeId ? { ...k, ...updates } : k
      ));
    });

    set({ clipKeyframes: newMap });
    invalidateCache();
  },

  moveKeyframe: (keyframeId, newTime) => {
    const { clipKeyframes, clips, invalidateCache } = get();
    const newMap = new Map<string, Keyframe[]>();

    clipKeyframes.forEach((keyframes, clipId) => {
      const clip = clips.find(c => c.id === clipId);
      const maxTime = clip?.duration ?? 999;

      newMap.set(clipId, keyframes.map(k => {
        if (k.id !== keyframeId) return k;
        return { ...k, time: Math.max(0, Math.min(newTime, maxTime)) };
      }).sort((a, b) => a.time - b.time));
    });

    set({ clipKeyframes: newMap });
    invalidateCache();
  },

  getClipKeyframes: (clipId) => {
    const { clipKeyframes } = get();
    return clipKeyframes.get(clipId) || [];
  },

  getInterpolatedTransform: (clipId, clipLocalTime) => {
    const { clips, clipKeyframes, playheadPosition } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) {
      return { ...DEFAULT_TRANSFORM };
    }

    // Ensure clip.transform exists and has all properties (handles loaded compositions with incomplete data)
    const baseTransform: ClipTransform = {
      opacity: clip.transform?.opacity ?? DEFAULT_TRANSFORM.opacity,
      blendMode: clip.transform?.blendMode ?? DEFAULT_TRANSFORM.blendMode,
      position: {
        x: clip.transform?.position?.x ?? DEFAULT_TRANSFORM.position.x,
        y: clip.transform?.position?.y ?? DEFAULT_TRANSFORM.position.y,
        z: clip.transform?.position?.z ?? DEFAULT_TRANSFORM.position.z,
      },
      scale: {
        x: clip.transform?.scale?.x ?? DEFAULT_TRANSFORM.scale.x,
        y: clip.transform?.scale?.y ?? DEFAULT_TRANSFORM.scale.y,
      },
      rotation: {
        x: clip.transform?.rotation?.x ?? DEFAULT_TRANSFORM.rotation.x,
        y: clip.transform?.rotation?.y ?? DEFAULT_TRANSFORM.rotation.y,
        z: clip.transform?.rotation?.z ?? DEFAULT_TRANSFORM.rotation.z,
      },
    };

    // Get this clip's own transform (with keyframe interpolation)
    const keyframes = clipKeyframes.get(clipId) || [];
    const ownTransform = keyframes.length === 0
      ? baseTransform
      : getInterpolatedClipTransform(keyframes, clipLocalTime, baseTransform);

    // If clip has a parent, compose with parent's transform
    if (clip.parentClipId) {
      const parentClip = clips.find(c => c.id === clip.parentClipId);
      if (parentClip) {
        // Calculate parent's local time based on current playhead position
        const parentLocalTime = playheadPosition - parentClip.startTime;
        // Recursively get parent's composed transform (handles nested parenting)
        const parentTransform = get().getInterpolatedTransform(clip.parentClipId, parentLocalTime);
        return composeTransforms(parentTransform, ownTransform);
      }
    }

    return ownTransform;
  },

  getInterpolatedEffects: (clipId, clipLocalTime) => {
    const { clips, clipKeyframes } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip || !clip.effects) {
      return [];
    }

    const keyframes = clipKeyframes.get(clipId) || [];
    if (keyframes.length === 0) {
      return clip.effects;
    }

    // Filter keyframes that are effect keyframes
    const effectKeyframes = keyframes.filter(k => k.property.startsWith('effect.'));

    if (effectKeyframes.length === 0) {
      return clip.effects;
    }

    // Clone effects and apply interpolated values
    return clip.effects.map(effect => {
      const newParams = { ...effect.params };

      // Check each numeric parameter for keyframes
      Object.keys(effect.params).forEach(paramName => {
        if (typeof effect.params[paramName] !== 'number') return;

        const propertyKey = `effect.${effect.id}.${paramName}`;
        const paramKeyframes = effectKeyframes.filter(k => k.property === propertyKey);

        if (paramKeyframes.length > 0) {
          // Interpolate the value
          newParams[paramName] = interpolateKeyframes(
            keyframes,
            propertyKey as AnimatableProperty,
            clipLocalTime,
            effect.params[paramName] as number
          );
        }
      });

      return { ...effect, params: newParams };
    });
  },

  getInterpolatedSpeed: (clipId, clipLocalTime) => {
    const { clips, clipKeyframes } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) return 1;

    const keyframes = clipKeyframes.get(clipId) || [];
    const defaultSpeed = clip.speed ?? 1;

    return getSpeedAtTime(keyframes, clipLocalTime, defaultSpeed);
  },

  getSourceTimeForClip: (clipId, clipLocalTime) => {
    const { clips, clipKeyframes } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) return clipLocalTime;

    const keyframes = clipKeyframes.get(clipId) || [];
    const defaultSpeed = clip.speed ?? 1;

    // If no speed keyframes, use simple multiplication
    const speedKeyframes = keyframes.filter(k => k.property === 'speed');
    if (speedKeyframes.length === 0 && defaultSpeed === 1) {
      return clipLocalTime;
    }

    // Calculate integrated source time
    const sourceTime = calculateSourceTime(keyframes, clipLocalTime, defaultSpeed);

    // Handle negative source time (reverse playback)
    return sourceTime;
  },

  hasKeyframes: (clipId, property) => {
    const { clipKeyframes } = get();
    const keyframes = clipKeyframes.get(clipId) || [];
    if (keyframes.length === 0) return false;
    if (!property) return true;
    return hasKeyframesForProperty(keyframes, property);
  },

  // Keyframe recording mode
  toggleKeyframeRecording: (clipId, property) => {
    const { keyframeRecordingEnabled } = get();
    const key = `${clipId}:${property}`;
    const newSet = new Set(keyframeRecordingEnabled);

    if (newSet.has(key)) {
      newSet.delete(key);
    } else {
      newSet.add(key);
    }

    set({ keyframeRecordingEnabled: newSet });
  },

  isRecording: (clipId, property) => {
    const { keyframeRecordingEnabled } = get();
    return keyframeRecordingEnabled.has(`${clipId}:${property}`);
  },

  setPropertyValue: (clipId, property, value) => {
    const { isRecording, addKeyframe, updateClipTransform, updateClipEffect, clips, hasKeyframes } = get();

    // Check if this property has keyframes (whether recording or not)
    const propertyHasKeyframes = hasKeyframes(clipId, property);

    if (isRecording(clipId, property) || propertyHasKeyframes) {
      // Recording mode OR property already has keyframes - create/update keyframe
      addKeyframe(clipId, property, value);
      // Also update clip.speed and recalculate duration
      if (property === 'speed') {
        const { invalidateCache, clipKeyframes, updateDuration } = get();
        const clip = clips.find(c => c.id === clipId);
        if (clip) {
          const keyframes = clipKeyframes.get(clipId) || [];
          const sourceDuration = clip.outPoint - clip.inPoint;
          const newDuration = calculateTimelineDuration(keyframes, sourceDuration, value);
          set({
            clips: clips.map(c => c.id === clipId ? { ...c, speed: value, duration: newDuration } : c)
          });
          updateDuration(); // Update timeline duration
        }
        invalidateCache();
      }
    } else {
      // Not recording and no keyframes - update static value
      const clip = clips.find(c => c.id === clipId);
      if (!clip) return;

      // Handle effect properties (format: effect.{effectId}.{paramName})
      if (property.startsWith('effect.')) {
        const parts = property.split('.');
        if (parts.length === 3) {
          const effectId = parts[1];
          const paramName = parts[2];
          updateClipEffect(clipId, effectId, { [paramName]: value });
        }
        return;
      }

      // Handle speed property (directly on clip, not transform)
      if (property === 'speed') {
        const { invalidateCache, updateDuration } = get();
        const sourceDuration = clip.outPoint - clip.inPoint;
        // For constant speed (no keyframes): duration = sourceDuration / |speed|
        const absSpeed = Math.abs(value) || 0.01; // Avoid division by zero
        const newDuration = sourceDuration / absSpeed;
        set({
          clips: clips.map(c => c.id === clipId ? { ...c, speed: value, duration: newDuration } : c)
        });
        updateDuration(); // Update timeline duration
        invalidateCache();
        return;
      }

      // Build partial transform update from property path
      const transformUpdate: Partial<ClipTransform> = {};

      if (property === 'opacity') {
        transformUpdate.opacity = value;
      } else if (property.startsWith('position.')) {
        const axis = property.split('.')[1] as 'x' | 'y' | 'z';
        transformUpdate.position = { ...clip.transform.position, [axis]: value };
      } else if (property.startsWith('scale.')) {
        const axis = property.split('.')[1] as 'x' | 'y';
        transformUpdate.scale = { ...clip.transform.scale, [axis]: value };
      } else if (property.startsWith('rotation.')) {
        const axis = property.split('.')[1] as 'x' | 'y' | 'z';
        transformUpdate.rotation = { ...clip.transform.rotation, [axis]: value };
      }

      updateClipTransform(clipId, transformUpdate);
    }
  },

  // Keyframe UI state - Track-based expansion
  toggleTrackExpanded: (trackId) => {
    const { expandedTracks } = get();
    const newSet = new Set(expandedTracks);

    if (newSet.has(trackId)) {
      newSet.delete(trackId);
    } else {
      newSet.add(trackId);
    }

    set({ expandedTracks: newSet });
  },

  isTrackExpanded: (trackId) => {
    const { expandedTracks } = get();
    return expandedTracks.has(trackId);
  },

  toggleTrackPropertyGroupExpanded: (trackId, groupName) => {
    const { expandedTrackPropertyGroups } = get();
    const newMap = new Map(expandedTrackPropertyGroups);
    const trackGroups = newMap.get(trackId) || new Set<string>();
    const newTrackGroups = new Set(trackGroups);

    if (newTrackGroups.has(groupName)) {
      newTrackGroups.delete(groupName);
    } else {
      newTrackGroups.add(groupName);
    }

    newMap.set(trackId, newTrackGroups);
    set({ expandedTrackPropertyGroups: newMap });
  },

  isTrackPropertyGroupExpanded: (trackId, groupName) => {
    const { expandedTrackPropertyGroups } = get();
    const trackGroups = expandedTrackPropertyGroups.get(trackId);
    return trackGroups?.has(groupName) ?? false;
  },

  // Calculate expanded track height based on visible property rows
  getExpandedTrackHeight: (trackId, baseHeight) => {
    const { expandedTracks, expandedCurveProperties, clips, selectedClipIds, clipKeyframes } = get();

    if (!expandedTracks.has(trackId)) {
      return baseHeight;
    }

    // Get the selected clip in this track
    const trackClips = clips.filter(c => c.trackId === trackId);
    const selectedTrackClip = trackClips.find(c => selectedClipIds.has(c.id));

    // If no clip is selected in this track, no property rows
    if (!selectedTrackClip) {
      return baseHeight;
    }

    const clipId = selectedTrackClip.id;
    const keyframes = clipKeyframes.get(clipId) || [];

    // If no keyframes at all, no property rows
    if (keyframes.length === 0) {
      return baseHeight;
    }

    // Flattened display: count unique properties with keyframes
    const uniqueProperties = new Set(keyframes.map(k => k.property));
    let extraHeight = uniqueProperties.size * PROPERTY_ROW_HEIGHT;

    // Add curve editor height for expanded properties
    const trackCurveProps = expandedCurveProperties.get(trackId);
    if (trackCurveProps) {
      trackCurveProps.forEach(prop => {
        if (uniqueProperties.has(prop)) {
          extraHeight += get().curveEditorHeight;
        }
      });
    }

    return baseHeight + extraHeight;
  },

  // Check if any clip on a track has keyframes
  trackHasKeyframes: (trackId) => {
    const { clips, clipKeyframes } = get();
    const trackClips = clips.filter(c => c.trackId === trackId);
    return trackClips.some(clip => {
      const kfs = clipKeyframes.get(clip.id);
      return kfs && kfs.length > 0;
    });
  },

  // Curve editor expansion
  toggleCurveExpanded: (trackId, property) => {
    const { expandedCurveProperties } = get();
    const isCurrentlyExpanded = expandedCurveProperties.get(trackId)?.has(property) ?? false;

    // Only one curve editor open at a time: close all, then open the new one
    const newMap = new Map<string, Set<AnimatableProperty>>();

    if (!isCurrentlyExpanded) {
      newMap.set(trackId, new Set([property]));
    }
    // If toggling off the currently open one, newMap stays empty (all closed)

    set({ expandedCurveProperties: newMap });
  },

  isCurveExpanded: (trackId, property) => {
    const { expandedCurveProperties } = get();
    const trackProps = expandedCurveProperties.get(trackId);
    return trackProps?.has(property) ?? false;
  },

  setCurveEditorHeight: (height) => {
    set({ curveEditorHeight: Math.round(Math.max(MIN_CURVE_EDITOR_HEIGHT, Math.min(MAX_CURVE_EDITOR_HEIGHT, height))) });
  },

  // Disable keyframes for a property: save current value as static, remove all keyframes, disable recording
  disablePropertyKeyframes: (clipId, property, currentValue) => {
    const { clips, clipKeyframes, keyframeRecordingEnabled, invalidateCache, updateClipTransform, updateClipEffect } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    // 1. Write current value to base clip value (same logic as setPropertyValue static path)
    if (property.startsWith('effect.')) {
      const parts = property.split('.');
      if (parts.length === 3) {
        const effectId = parts[1];
        const paramName = parts[2];
        updateClipEffect(clipId, effectId, { [paramName]: currentValue });
      }
    } else if (property === 'speed') {
      const { updateDuration } = get();
      const sourceDuration = clip.outPoint - clip.inPoint;
      const absSpeed = Math.abs(currentValue) || 0.01;
      const newDuration = sourceDuration / absSpeed;
      set({
        clips: get().clips.map(c => c.id === clipId ? { ...c, speed: currentValue, duration: newDuration } : c)
      });
      updateDuration();
    } else if (property === 'opacity') {
      updateClipTransform(clipId, { opacity: currentValue });
    } else if (property.startsWith('position.')) {
      const axis = property.split('.')[1] as 'x' | 'y' | 'z';
      updateClipTransform(clipId, { position: { ...clip.transform.position, [axis]: currentValue } });
    } else if (property.startsWith('scale.')) {
      const axis = property.split('.')[1] as 'x' | 'y';
      updateClipTransform(clipId, { scale: { ...clip.transform.scale, [axis]: currentValue } });
    } else if (property.startsWith('rotation.')) {
      const axis = property.split('.')[1] as 'x' | 'y' | 'z';
      updateClipTransform(clipId, { rotation: { ...clip.transform.rotation, [axis]: currentValue } });
    }

    // 2. Remove all keyframes for this property
    const existingKeyframes = clipKeyframes.get(clipId) || [];
    const filtered = existingKeyframes.filter(k => k.property !== property);
    const newMap = new Map(clipKeyframes);
    if (filtered.length > 0) {
      newMap.set(clipId, filtered);
    } else {
      newMap.delete(clipId);
    }

    // 3. Disable recording
    const newRecording = new Set(keyframeRecordingEnabled);
    newRecording.delete(`${clipId}:${property}`);

    set({ clipKeyframes: newMap, keyframeRecordingEnabled: newRecording });
    invalidateCache();
  },

  // Bezier handle manipulation
  updateBezierHandle: (keyframeId, handle, position) => {
    const { clipKeyframes, invalidateCache } = get();
    const newMap = new Map<string, Keyframe[]>();

    clipKeyframes.forEach((keyframes, clipId) => {
      newMap.set(clipId, keyframes.map(k => {
        if (k.id !== keyframeId) return k;
        return {
          ...k,
          easing: 'bezier' as const,
          [handle === 'in' ? 'handleIn' : 'handleOut']: position,
        };
      }));
    });

    set({ clipKeyframes: newMap });
    invalidateCache();
  },
});
