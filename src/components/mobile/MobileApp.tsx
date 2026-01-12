// Mobile App - Root component for mobile UI

import { useState, useCallback } from 'react';
import { MobilePreview } from './MobilePreview';
import { MobileTimeline } from './MobileTimeline';
import { MobileToolbar } from './MobileToolbar';
import { MobilePropertiesPanel } from './MobilePropertiesPanel';
import { MobileMediaPanel } from './MobileMediaPanel';
import { MobileOptionsMenu } from './MobileOptionsMenu';
import { useGlobalHistory } from '../../hooks/useGlobalHistory';
import { useTimelineStore } from '../../stores/timeline';
import { undo, redo } from '../../stores/historyStore';
import './mobile.css';

export function MobileApp() {
  // Initialize global undo/redo system
  useGlobalHistory();

  // Panel states
  const [propertiesPanelOpen, setPropertiesPanelOpen] = useState(false);
  const [mediaPanelOpen, setMediaPanelOpen] = useState(false);
  const [optionsMenuOpen, setOptionsMenuOpen] = useState(false);

  // Active slider state (when user taps a control in properties)
  const [activeSlider, setActiveSlider] = useState<{
    property: string;
    label: string;
    min: number;
    max: number;
    step: number;
    value: number;
  } | null>(null);

  // Precision mode for playhead
  const [precisionMode, setPrecisionMode] = useState(false);

  // Timeline store for cut action
  const splitClipAtPlayhead = useTimelineStore((s) => s.splitClipAtPlayhead);

  // Handle cut at playhead
  const handleCut = useCallback(() => {
    splitClipAtPlayhead();
  }, [splitClipAtPlayhead]);

  // Handle slider activation from properties panel
  const handleActivateSlider = useCallback((slider: typeof activeSlider) => {
    setActiveSlider(slider);
    setPropertiesPanelOpen(false);
  }, []);

  // Handle slider close
  const handleCloseSlider = useCallback(() => {
    setActiveSlider(null);
  }, []);

  // Two-finger swipe for undo/redo
  const handleTwoFingerSwipe = useCallback((direction: 'left' | 'right') => {
    if (direction === 'left') {
      undo();
    } else {
      redo();
    }
  }, []);

  return (
    <div
      className="mobile-app"
      onTouchStart={(e) => {
        // Track two-finger swipes
        if (e.touches.length === 2) {
          const startX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
          (e.currentTarget as any)._twoFingerStartX = startX;
        }
      }}
      onTouchEnd={(e) => {
        const startX = (e.currentTarget as any)._twoFingerStartX;
        if (startX !== undefined && e.changedTouches.length > 0) {
          const endX = e.changedTouches[0].clientX;
          const delta = endX - startX;
          if (Math.abs(delta) > 50) {
            handleTwoFingerSwipe(delta < 0 ? 'left' : 'right');
          }
          delete (e.currentTarget as any)._twoFingerStartX;
        }
      }}
    >
      {/* Preview - always visible */}
      <div className="mobile-preview-container">
        <MobilePreview />

        {/* Pull-down indicator */}
        <div
          className="mobile-pull-indicator"
          onClick={() => setPropertiesPanelOpen(true)}
        >
          <div className="pull-handle" />
        </div>

        {/* Side menu buttons */}
        <button
          className="mobile-menu-btn left"
          onClick={() => setMediaPanelOpen(true)}
        >
          ☰
        </button>
        <button
          className="mobile-menu-btn right"
          onClick={() => setOptionsMenuOpen(true)}
        >
          ⚙
        </button>
      </div>

      {/* Active Slider - shows between preview and timeline */}
      {activeSlider && (
        <div className="mobile-active-slider" onClick={handleCloseSlider}>
          <div className="active-slider-content" onClick={(e) => e.stopPropagation()}>
            <span className="active-slider-label">{activeSlider.label}</span>
            <input
              type="range"
              min={activeSlider.min}
              max={activeSlider.max}
              step={activeSlider.step}
              value={activeSlider.value}
              onChange={(e) => {
                // Update value through timeline store
                const value = parseFloat(e.target.value);
                // This will be connected to actual property updates
                setActiveSlider({ ...activeSlider, value });
              }}
            />
            <span className="active-slider-value">{activeSlider.value.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Toolbar - Cut and Precision buttons */}
      <MobileToolbar
        onCut={handleCut}
        precisionMode={precisionMode}
        onPrecisionModeChange={setPrecisionMode}
      />

      {/* Timeline - bottom */}
      <div className="mobile-timeline-container">
        <MobileTimeline precisionMode={precisionMode} />
      </div>

      {/* Properties Panel - pull down from top */}
      <MobilePropertiesPanel
        isOpen={propertiesPanelOpen}
        onClose={() => setPropertiesPanelOpen(false)}
        onActivateSlider={handleActivateSlider}
      />

      {/* Media Panel - swipe from left */}
      <MobileMediaPanel
        isOpen={mediaPanelOpen}
        onClose={() => setMediaPanelOpen(false)}
      />

      {/* Options Menu - swipe from right */}
      <MobileOptionsMenu
        isOpen={optionsMenuOpen}
        onClose={() => setOptionsMenuOpen(false)}
      />

      {/* Edge swipe detectors */}
      <div
        className="mobile-edge-left"
        onTouchStart={(e) => {
          const touch = e.touches[0];
          if (touch.clientX < 20) {
            setMediaPanelOpen(true);
          }
        }}
      />
      <div
        className="mobile-edge-right"
        onTouchStart={(e) => {
          const touch = e.touches[0];
          const screenWidth = window.innerWidth;
          if (touch.clientX > screenWidth - 20) {
            setOptionsMenuOpen(true);
          }
        }}
      />
    </div>
  );
}
