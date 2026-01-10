// ImageCropper - Pan and zoom image editor for AI video frame input
// Shows exact crop that will be sent to the API

import { useState, useRef, useCallback, useEffect } from 'react';

interface ImageCropperProps {
  imageUrl: string | null;
  aspectRatio: { width: number; height: number };
  onClear: () => void;
  onCropChange: (cropData: CropData) => void;
  disabled?: boolean;
  label: string;
  onDropOrClick: () => void;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onUseCurrentFrame: () => void;
}

export interface CropData {
  offsetX: number;  // -1 to 1, 0 = centered
  offsetY: number;  // -1 to 1, 0 = centered
  scale: number;    // 1 = fit, >1 = zoomed in
}

export function ImageCropper({
  imageUrl,
  aspectRatio,
  onClear,
  onCropChange,
  disabled,
  label,
  onDropOrClick,
  onDrop,
  onUseCurrentFrame,
}: ImageCropperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Crop state
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [scale, setScale] = useState(1);

  // Image natural dimensions
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Track previous imageUrl to detect changes
  const prevImageUrlRef = useRef<string | null>(null);

  // Reset crop when image changes and load dimensions
  useEffect(() => {
    // Reset crop when image changes
    if (imageUrl !== prevImageUrlRef.current) {
      prevImageUrlRef.current = imageUrl;
      // Use setTimeout to batch state updates
      setTimeout(() => {
        setOffset({ x: 0, y: 0 });
        setScale(1);
      }, 0);
    }

    if (!imageUrl) {
      setImageDimensions({ width: 0, height: 0 });
      return;
    }

    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Notify parent of crop changes
  useEffect(() => {
    onCropChange({
      offsetX: offset.x,
      offsetY: offset.y,
      scale,
    });
  }, [offset, scale, onCropChange]);

  // Calculate image transform style
  const getImageStyle = useCallback(() => {
    if (!imageUrl || !imageDimensions.width) return {};

    const containerAspect = aspectRatio.width / aspectRatio.height;
    const imageAspect = imageDimensions.width / imageDimensions.height;

    // Calculate base scale to cover the container
    let baseScale: number;
    if (imageAspect > containerAspect) {
      // Image is wider - fit height, overflow width
      baseScale = 1;
    } else {
      // Image is taller - fit width, overflow height
      baseScale = containerAspect / imageAspect;
    }

    const totalScale = baseScale * scale;

    // Calculate max offset based on overflow
    const overflowX = Math.max(0, (totalScale * 100 - 100) / 2);
    const overflowY = Math.max(0, (totalScale * (100 / imageAspect * containerAspect) - 100) / 2);

    // Clamp offsets
    const clampedX = Math.max(-overflowX, Math.min(overflowX, offset.x * overflowX));
    const clampedY = Math.max(-overflowY, Math.min(overflowY, offset.y * overflowY));

    return {
      transform: `translate(${clampedX}%, ${clampedY}%) scale(${totalScale})`,
      transformOrigin: 'center center',
    };
  }, [imageUrl, imageDimensions, aspectRatio, offset, scale]);

  // Handle pointer lock for infinite dragging
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handlePointerLockChange = () => {
      const isLocked = document.pointerLockElement === container;
      setIsDragging(isLocked);
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== container) return;

      const rect = container.getBoundingClientRect();
      // Use movementX/Y for infinite movement
      const deltaX = e.movementX / rect.width;
      const deltaY = e.movementY / rect.height;

      setOffset(prev => ({
        x: Math.max(-1, Math.min(1, prev.x + deltaX * 2)),
        y: Math.max(-1, Math.min(1, prev.y + deltaY * 2)),
      }));
    };

    document.addEventListener('pointerlockchange', handlePointerLockChange);
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('pointerlockchange', handlePointerLockChange);
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  // Handle mouse down for drag - request pointer lock
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (disabled || !imageUrl) return;
    if (e.button !== 0) return; // Only left click

    e.preventDefault();
    containerRef.current?.requestPointerLock();
  }, [disabled, imageUrl]);

  // Handle mouse up - release pointer lock
  const handleMouseUp = useCallback(() => {
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }
  }, []);

  // Handle wheel for zoom - use native event to properly prevent scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      if (disabled || !imageUrl) return;

      // Prevent page scroll
      e.preventDefault();
      e.stopPropagation();

      // Smooth, slow zoom
      const zoomFactor = 0.001;
      const delta = -e.deltaY * zoomFactor;

      setScale(prev => Math.max(1, Math.min(3, prev + delta)));
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [disabled, imageUrl]);

  // Handle drag over for file drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  // Handle click to open file picker
  const handleClick = useCallback(() => {
    if (!imageUrl) {
      onDropOrClick();
    }
  }, [imageUrl, onDropOrClick]);

  return (
    <div className="image-cropper-group">
      <label>{label}</label>
      <div
        ref={containerRef}
        className={`image-cropper ${imageUrl ? 'has-image' : ''} ${isDragging ? 'dragging' : ''}`}
        style={{ aspectRatio: `${aspectRatio.width} / ${aspectRatio.height}` }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onDragOver={handleDragOver}
        onDrop={onDrop}
        onClick={handleClick}
      >
        {imageUrl ? (
          <>
            <div className="image-cropper-viewport">
              <img
                src={imageUrl}
                alt={label}
                style={getImageStyle()}
                draggable={false}
              />
            </div>
            <button
              className="clear-image"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
            >
              ×
            </button>
            {scale > 1 && (
              <div className="zoom-indicator">{Math.round(scale * 100)}%</div>
            )}
            <div className="crop-hint">Drag to pan • Scroll to zoom</div>
          </>
        ) : (
          <span className="drop-hint">Drop or click</span>
        )}
      </div>
      <button
        className="btn-use-current"
        onClick={onUseCurrentFrame}
        disabled={disabled}
      >
        Use Current Frame
      </button>
    </div>
  );
}

// Export cropped image as data URL
export async function exportCroppedImage(
  imageUrl: string,
  cropData: CropData,
  aspectRatio: { width: number; height: number },
  outputWidth: number = 1280
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const canvas = document.createElement('canvas');
      const outputHeight = Math.round(outputWidth / aspectRatio.width * aspectRatio.height);
      canvas.width = outputWidth;
      canvas.height = outputHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Failed to get canvas context'));
        return;
      }

      const imgAspect = img.naturalWidth / img.naturalHeight;
      const targetAspect = aspectRatio.width / aspectRatio.height;

      // Calculate source rectangle
      let srcWidth: number, srcHeight: number, srcX: number, srcY: number;

      if (imgAspect > targetAspect) {
        // Image is wider - crop sides
        srcHeight = img.naturalHeight / cropData.scale;
        srcWidth = srcHeight * targetAspect;
      } else {
        // Image is taller - crop top/bottom
        srcWidth = img.naturalWidth / cropData.scale;
        srcHeight = srcWidth / targetAspect;
      }

      // Apply offset
      const maxOffsetX = (img.naturalWidth - srcWidth) / 2;
      const maxOffsetY = (img.naturalHeight - srcHeight) / 2;

      srcX = (img.naturalWidth - srcWidth) / 2 - cropData.offsetX * maxOffsetX;
      srcY = (img.naturalHeight - srcHeight) / 2 - cropData.offsetY * maxOffsetY;

      // Clamp source rectangle
      srcX = Math.max(0, Math.min(img.naturalWidth - srcWidth, srcX));
      srcY = Math.max(0, Math.min(img.naturalHeight - srcHeight, srcY));

      // Draw cropped image
      ctx.drawImage(
        img,
        srcX, srcY, srcWidth, srcHeight,
        0, 0, outputWidth, outputHeight
      );

      resolve(canvas.toDataURL('image/png'));
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = imageUrl;
  });
}
