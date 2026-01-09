// PickWhip component - After Effects-style pick whip for clip parenting
// Drag from this icon to another clip to set it as parent

import { useCallback, useRef, useState, useEffect } from 'react';

interface PickWhipProps {
  clipId: string;
  clipName: string;
  parentClipId: string | undefined;
  parentClipName: string | undefined;
  onSetParent: (clipId: string, parentClipId: string | null) => void;
  onDragStart: (clipId: string, startX: number, startY: number) => void;
  onDragEnd: () => void;
}

export function PickWhip({
  clipId,
  clipName,
  parentClipId,
  parentClipName,
  onSetParent,
  onDragStart,
  onDragEnd,
}: PickWhipProps) {
  const iconRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);

    const rect = iconRef.current?.getBoundingClientRect();
    if (rect) {
      onDragStart(clipId, rect.left + rect.width / 2, rect.top + rect.height / 2);
    }
  }, [clipId, onDragStart]);

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      onDragEnd();
    }
  }, [isDragging, onDragEnd]);

  // Handle click to unparent
  const handleClick = useCallback((e: React.MouseEvent) => {
    if (parentClipId && !isDragging) {
      e.preventDefault();
      e.stopPropagation();
      onSetParent(clipId, null);
    }
  }, [clipId, parentClipId, isDragging, onSetParent]);

  // Add global mouse up listener when dragging
  useEffect(() => {
    if (isDragging) {
      const handleGlobalMouseUp = () => {
        setIsDragging(false);
        onDragEnd();
      };

      window.addEventListener('mouseup', handleGlobalMouseUp);
      return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
    }
  }, [isDragging, onDragEnd]);

  const hasParent = !!parentClipId;

  return (
    <div
      ref={iconRef}
      className={`pick-whip ${isDragging ? 'dragging' : ''} ${hasParent ? 'has-parent' : ''}`}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
      title={
        hasParent
          ? `Parented to: ${parentClipName || parentClipId}\nClick to unparent, or drag to new parent`
          : `Drag to parent clip "${clipName}" to another clip`
      }
    >
      {/* Pick Whip / Spiral Icon */}
      <svg viewBox="0 0 16 16" width="14" height="14" className="pick-whip-icon">
        {hasParent ? (
          // Connected state - solid spiral with dot
          <>
            <path
              d="M8 2C5 2 3 4 3 7s2 5 5 5c2 0 3.5-1 4-2.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <circle cx="12" cy="9.5" r="2" fill="currentColor" />
          </>
        ) : (
          // Unconnected state - outline only
          <>
            <path
              d="M8 2C5 2 3 4 3 7s2 5 5 5c2 0 3.5-1 4-2.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeOpacity="0.5"
            />
            <circle cx="12" cy="9.5" r="1.5" fill="none" stroke="currentColor" strokeOpacity="0.5" />
          </>
        )}
      </svg>
    </div>
  );
}
