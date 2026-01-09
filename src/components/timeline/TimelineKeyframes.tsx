// TimelineKeyframes component - Keyframe diamonds/handles with drag support

import { memo, useMemo, useState, useEffect, useCallback } from 'react';
import type { TimelineKeyframesProps } from './types';
import type { EasingType, AnimatableProperty } from '../../types';

interface KeyframeData {
  id: string;
  clipId: string;
  time: number;
  property: AnimatableProperty;
  value: number;
  easing: string;
}

interface KeyframeDisplay {
  kf: KeyframeData;
  clip: TimelineKeyframesProps['clips'][0];
  absTime: number;
}

// Easing options for context menu
const EASING_OPTIONS: { value: EasingType; label: string }[] = [
  { value: 'linear', label: 'Linear' },
  { value: 'ease-in', label: 'Ease In' },
  { value: 'ease-out', label: 'Ease Out' },
  { value: 'ease-in-out', label: 'Ease In-Out' },
];

function TimelineKeyframesComponent({
  trackId,
  property,
  clips,
  selectedKeyframeIds,
  clipKeyframes,
  onSelectKeyframe,
  onMoveKeyframe,
  onUpdateKeyframe,
  timeToPixel,
  pixelToTime,
}: TimelineKeyframesProps) {
  // Drag state
  const [dragState, setDragState] = useState<{
    keyframeId: string;
    clipId: string;
    startX: number;
    startTime: number;
    clipStartTime: number;
  } | null>(null);

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    keyframeId: string;
    currentEasing: string;
  } | null>(null);

  // Get all clips on this track
  const trackClips = useMemo(
    () => clips.filter((c) => c.trackId === trackId),
    [clips, trackId]
  );

  // Get all keyframes once and group by clip/property
  // Now depends on clipKeyframes Map directly for proper reactivity
  const allKeyframes = useMemo(() => {
    const result: KeyframeDisplay[] = [];

    trackClips.forEach((clip) => {
      const kfs = clipKeyframes.get(clip.id) || [];
      kfs
        .filter((k) => k.property === property)
        .forEach((kf) => {
          result.push({
            kf,
            clip,
            absTime: clip.startTime + kf.time,
          });
        });
    });

    return result;
  }, [trackClips, property, clipKeyframes]);

  // Handle keyframe drag
  const handleMouseDown = useCallback((
    e: React.MouseEvent,
    kf: KeyframeDisplay['kf'],
    clip: KeyframeDisplay['clip']
  ) => {
    if (e.button !== 0) return; // Left click only
    e.preventDefault();
    e.stopPropagation();

    // Select the keyframe
    onSelectKeyframe(kf.id, e.shiftKey);

    // Start drag
    setDragState({
      keyframeId: kf.id,
      clipId: clip.id,
      startX: e.clientX,
      startTime: kf.time,
      clipStartTime: clip.startTime,
    });
  }, [onSelectKeyframe]);

  // Handle drag movement
  useEffect(() => {
    if (!dragState) return;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - dragState.startX;

      // Shift for finer movement (10x slower)
      const sensitivity = e.shiftKey ? 0.1 : 1;
      const effectiveDelta = deltaX * sensitivity;

      // Convert pixel delta to time delta
      const currentPixel = timeToPixel(dragState.clipStartTime + dragState.startTime);
      const newPixel = currentPixel + effectiveDelta;
      const newAbsTime = pixelToTime(newPixel);

      // Calculate new clip-local time (ensure it stays within clip bounds)
      const clip = clips.find(c => c.id === dragState.clipId);
      if (!clip) return;

      const newClipTime = newAbsTime - clip.startTime;
      const clampedTime = Math.max(0, Math.min(clip.duration, newClipTime));

      onMoveKeyframe(dragState.keyframeId, clampedTime);
    };

    const handleMouseUp = () => {
      setDragState(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragState, timeToPixel, pixelToTime, clips, onMoveKeyframe]);

  // Handle right-click context menu
  const handleContextMenu = useCallback((
    e: React.MouseEvent,
    kf: KeyframeDisplay['kf']
  ) => {
    e.preventDefault();
    e.stopPropagation();

    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      keyframeId: kf.id,
      currentEasing: kf.easing,
    });
  }, []);

  // Close context menu on click outside
  useEffect(() => {
    if (!contextMenu) return;

    const handleClick = () => setContextMenu(null);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [contextMenu]);

  // Handle easing selection
  const handleEasingSelect = useCallback((easing: EasingType) => {
    if (contextMenu) {
      onUpdateKeyframe(contextMenu.keyframeId, { easing });
      setContextMenu(null);
    }
  }, [contextMenu, onUpdateKeyframe]);

  return (
    <>
      {allKeyframes.map(({ kf, clip, absTime }) => {
        const xPos = timeToPixel(absTime);
        const isSelected = selectedKeyframeIds.has(kf.id);
        const isDragging = dragState?.keyframeId === kf.id;

        return (
          <div
            key={kf.id}
            className={`keyframe-diamond ${isSelected ? 'selected' : ''} ${isDragging ? 'dragging' : ''}`}
            style={{ left: `${xPos}px` }}
            onMouseDown={(e) => handleMouseDown(e, kf, clip)}
            onContextMenu={(e) => handleContextMenu(e, kf)}
            title={`${property}: ${kf.value.toFixed(3)} @ ${absTime.toFixed(2)}s\nEasing: ${kf.easing}\nDrag to move (Shift for fine control)\nRight-click to change easing`}
          />
        );
      })}

      {/* Context Menu for easing selection */}
      {contextMenu && (
        <div
          className="keyframe-context-menu"
          style={{
            position: 'fixed',
            left: contextMenu.x,
            top: contextMenu.y,
            zIndex: 10000,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="context-menu-title">Easing</div>
          {EASING_OPTIONS.map((option) => (
            <div
              key={option.value}
              className={`context-menu-item ${contextMenu.currentEasing === option.value ? 'active' : ''}`}
              onClick={() => handleEasingSelect(option.value)}
            >
              {option.label}
              {contextMenu.currentEasing === option.value && <span className="checkmark">✓</span>}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export const TimelineKeyframes = memo(TimelineKeyframesComponent);
