import { useState, useEffect, useCallback, useRef } from 'react';
import { useDockStore } from '../../stores/dockStore';
import type { PanelType } from '../../types/dock';

function ClippyMascot() {
  const [useWebP, setUseWebP] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const source = videoRef.current?.querySelector('source');
    if (source) {
      source.addEventListener('error', () => setUseWebP(true), { once: true });
    }
  }, []);

  if (useWebP) {
    return <img src="/clippy.webp" alt="" className="tutorial-clippy" draggable={false} />;
  }

  return (
    <video
      ref={videoRef}
      className="tutorial-clippy"
      autoPlay
      loop
      muted
      playsInline
      disablePictureInPicture
    >
      <source src="/clippy.webm" type="video/webm" />
    </video>
  );
}

// Part 1: Panel-level steps (existing)
interface PanelStep {
  groupId: string;
  panelType: PanelType;
  title: string;
  description: string;
  tooltipPosition: 'top' | 'bottom' | 'left' | 'right';
}

const PANEL_STEPS: PanelStep[] = [
  {
    groupId: 'timeline-group',
    panelType: 'timeline',
    title: 'Timeline',
    description: 'Arrange and edit your clips on tracks. Drag to move, trim edges, add keyframes and transitions.',
    tooltipPosition: 'top',
  },
  {
    groupId: 'preview-group',
    panelType: 'preview',
    title: 'Preview',
    description: 'Live preview of your composition. Play, pause, and scrub through your project in real-time.',
    tooltipPosition: 'left',
  },
  {
    groupId: 'left-group',
    panelType: 'media',
    title: 'Media',
    description: 'Import and organize your media files. Drag clips from here onto the Timeline to start editing.',
    tooltipPosition: 'right',
  },
  {
    groupId: 'right-group',
    panelType: 'clip-properties',
    title: 'Properties',
    description: 'Adjust transforms, effects, and masks for the selected clip. Select a clip in the Timeline to get started.',
    tooltipPosition: 'left',
  },
];

// Part 2: Timeline element-level steps
interface TimelineStep {
  selector: string;
  title: string;
  description: string;
  tooltipPosition: 'top' | 'bottom' | 'left' | 'right';
}

const TIMELINE_STEPS: TimelineStep[] = [
  {
    selector: '.timeline-controls',
    title: 'Playback',
    description: 'Play, Stop und Loop — steuere die Wiedergabe deiner Composition.',
    tooltipPosition: 'bottom',
  },
  {
    selector: '.timeline-time',
    title: 'Timecode',
    description: 'Aktuelle Position und Gesamtdauer. Klicke auf die Dauer um sie zu ändern.',
    tooltipPosition: 'bottom',
  },
  {
    selector: '.timeline-zoom',
    title: 'Tools & Zoom',
    description: 'Snapping, Cut-Tool, Zoom und Fit — kontrolliere die Timeline-Ansicht.',
    tooltipPosition: 'bottom',
  },
  {
    selector: '.timeline-inout-controls',
    title: 'In/Out Points',
    description: 'Setze In- (I) und Out-Punkte (O) um den Export-Bereich festzulegen.',
    tooltipPosition: 'bottom',
  },
  {
    selector: '.timeline-tracks-controls',
    title: 'Tracks',
    description: 'Füge Video-, Audio- oder Text-Tracks hinzu.',
    tooltipPosition: 'bottom',
  },
  {
    selector: '.timeline-navigator',
    title: 'Navigator',
    description: 'Scrolle und zoome die Timeline. Ziehe die Kanten um hinein/herauszuzoomen.',
    tooltipPosition: 'top',
  },
];

const TOOLTIP_GAP = 16;

interface Props {
  onClose: () => void;
  part?: 1 | 2;
}

export function TutorialOverlay({ onClose, part = 1 }: Props) {
  const [stepIndex, setStepIndex] = useState(0);
  const [panelRect, setPanelRect] = useState<DOMRect | null>(null);
  const [highlightRect, setHighlightRect] = useState<DOMRect | null>(null);
  const [isClosing, setIsClosing] = useState(false);
  const activatePanelType = useDockStore((s) => s.activatePanelType);
  const closingRef = useRef(false);

  const isPart2 = part === 2;
  const steps = isPart2 ? TIMELINE_STEPS : PANEL_STEPS;
  const step = steps[stepIndex];

  // Find and measure targets
  const measureTargets = useCallback(() => {
    if (isPart2) {
      // Part 2: Always measure the timeline panel for the SVG mask cutout
      const timelineEl = document.querySelector('[data-group-id="timeline-group"]');
      if (timelineEl) {
        setPanelRect(timelineEl.getBoundingClientRect());
      } else {
        setPanelRect(null);
      }
      // Measure the specific element for the highlight ring
      const targetEl = document.querySelector((step as TimelineStep).selector);
      if (targetEl) {
        setHighlightRect(targetEl.getBoundingClientRect());
      } else {
        setHighlightRect(null);
      }
    } else {
      // Part 1: Measure the panel group
      const el = document.querySelector(`[data-group-id="${(step as PanelStep).groupId}"]`);
      if (el) {
        setPanelRect(el.getBoundingClientRect());
      } else {
        setPanelRect(null);
      }
      setHighlightRect(null);
    }
  }, [isPart2, step]);

  // Activate the correct tab and measure on step change
  useEffect(() => {
    if (!isPart2) {
      activatePanelType((step as PanelStep).panelType);
    }
    // Small delay to let tab switch render before measuring
    const timer = setTimeout(measureTargets, 50);
    return () => clearTimeout(timer);
  }, [step, isPart2, activatePanelType, measureTargets]);

  // Re-measure on resize
  useEffect(() => {
    window.addEventListener('resize', measureTargets);
    return () => window.removeEventListener('resize', measureTargets);
  }, [measureTargets]);

  const close = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setIsClosing(true);
    setTimeout(onClose, 200);
  }, [onClose]);

  const advance = useCallback(() => {
    if (isClosing) return;
    if (stepIndex < steps.length - 1) {
      setStepIndex(stepIndex + 1);
    } else {
      close();
    }
  }, [stepIndex, steps.length, isClosing, close]);

  // Escape to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [close]);

  // Compute tooltip position clamped to viewport
  // For Part 2, position relative to the highlight ring; for Part 1, relative to the panel
  const getTooltipStyle = (): React.CSSProperties => {
    const anchorRect = isPart2 ? highlightRect : panelRect;
    if (!anchorRect) return { opacity: 0 };

    const tooltipW = 380;
    const tooltipH = 180;
    const pos = step.tooltipPosition;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = 0;
    let top = 0;

    if (pos === 'top') {
      left = anchorRect.left + anchorRect.width / 2 - tooltipW / 2;
      top = anchorRect.top - tooltipH - TOOLTIP_GAP;
    } else if (pos === 'bottom') {
      left = anchorRect.left + anchorRect.width / 2 - tooltipW / 2;
      top = anchorRect.bottom + TOOLTIP_GAP;
    } else if (pos === 'left') {
      left = anchorRect.left - tooltipW - TOOLTIP_GAP;
      top = anchorRect.top + anchorRect.height / 2 - tooltipH / 2;
    } else if (pos === 'right') {
      left = anchorRect.right + TOOLTIP_GAP;
      top = anchorRect.top + anchorRect.height / 2 - tooltipH / 2;
    }

    // Clamp to viewport
    left = Math.max(12, Math.min(left, vw - tooltipW - 12));
    top = Math.max(12, Math.min(top, vh - tooltipH - 12));

    return { left, top };
  };

  return (
    <div
      className={`tutorial-backdrop ${isClosing ? 'closing' : ''}`}
      onClick={advance}
    >
      <svg className="tutorial-overlay-svg" width="100%" height="100%">
        <defs>
          <mask id="tutorial-mask">
            <rect width="100%" height="100%" fill="white" />
            {panelRect && (
              <rect
                x={panelRect.left}
                y={panelRect.top}
                width={panelRect.width}
                height={panelRect.height}
                rx="8"
                fill="black"
                style={{ transition: 'all 400ms ease' }}
              />
            )}
          </mask>
        </defs>
        <rect
          width="100%"
          height="100%"
          fill="rgba(0,0,0,0.75)"
          mask="url(#tutorial-mask)"
        />
      </svg>

      {/* Part 2: Yellow highlight ring over the target element */}
      {isPart2 && highlightRect && (
        <div className="tutorial-highlight-ring" style={{
          left: highlightRect.left - 4,
          top: highlightRect.top - 4,
          width: highlightRect.width + 8,
          height: highlightRect.height + 8,
        }} />
      )}

      <div className="tutorial-tooltip" style={getTooltipStyle()}>
        <div className={`tutorial-tooltip-arrow tutorial-tooltip-arrow--${step.tooltipPosition}`} />
        <ClippyMascot />
        <div className="tutorial-tooltip-content">
          <div className="tutorial-tooltip-text">
            <div className="tutorial-tooltip-step">Step {stepIndex + 1} of {steps.length}</div>
            <div className="tutorial-tooltip-title">{step.title}</div>
            <div className="tutorial-tooltip-desc">{step.description}</div>
            <div className="tutorial-dots">
              {steps.map((_: PanelStep | TimelineStep, i: number) => (
                <span
                  key={i}
                  className={`tutorial-dot ${i === stepIndex ? 'active' : ''} ${i < stepIndex ? 'completed' : ''}`}
                />
              ))}
            </div>
            <div className="tutorial-tooltip-hint">
              {stepIndex < steps.length - 1 ? 'Click anywhere to continue' : 'Click to finish'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
