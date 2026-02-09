import { useState, useEffect, useCallback, useRef } from 'react';
import { useDockStore } from '../../stores/dockStore';
import type { PanelType } from '../../types/dock';

interface TutorialStep {
  groupId: string;
  panelType: PanelType;
  title: string;
  description: string;
  tooltipPosition: 'top' | 'bottom' | 'left' | 'right';
}

const TUTORIAL_STEPS: TutorialStep[] = [
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

const TOOLTIP_GAP = 16;

interface Props {
  onClose: () => void;
}

export function TutorialOverlay({ onClose }: Props) {
  const [stepIndex, setStepIndex] = useState(0);
  const [panelRect, setPanelRect] = useState<DOMRect | null>(null);
  const [isClosing, setIsClosing] = useState(false);
  const activatePanelType = useDockStore((s) => s.activatePanelType);
  const closingRef = useRef(false);

  const step = TUTORIAL_STEPS[stepIndex];

  // Find and measure the target panel
  const measurePanel = useCallback(() => {
    const el = document.querySelector(`[data-group-id="${step.groupId}"]`);
    if (el) {
      setPanelRect(el.getBoundingClientRect());
    } else {
      setPanelRect(null);
    }
  }, [step.groupId]);

  // Activate the correct tab and measure on step change
  useEffect(() => {
    activatePanelType(step.panelType);
    // Small delay to let tab switch render before measuring
    const timer = setTimeout(measurePanel, 50);
    return () => clearTimeout(timer);
  }, [step, activatePanelType, measurePanel]);

  // Re-measure on resize
  useEffect(() => {
    window.addEventListener('resize', measurePanel);
    return () => window.removeEventListener('resize', measurePanel);
  }, [measurePanel]);

  const close = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setIsClosing(true);
    setTimeout(onClose, 200);
  }, [onClose]);

  const advance = useCallback(() => {
    if (isClosing) return;
    if (stepIndex < TUTORIAL_STEPS.length - 1) {
      setStepIndex(stepIndex + 1);
    } else {
      close();
    }
  }, [stepIndex, isClosing, close]);

  // Escape to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [close]);

  // Compute tooltip position clamped to viewport
  const getTooltipStyle = (): React.CSSProperties => {
    if (!panelRect) return { opacity: 0 };

    const tooltipW = 300;
    const tooltipH = 160;
    const pos = step.tooltipPosition;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = 0;
    let top = 0;

    if (pos === 'top') {
      left = panelRect.left + panelRect.width / 2 - tooltipW / 2;
      top = panelRect.top - tooltipH - TOOLTIP_GAP;
    } else if (pos === 'bottom') {
      left = panelRect.left + panelRect.width / 2 - tooltipW / 2;
      top = panelRect.bottom + TOOLTIP_GAP;
    } else if (pos === 'left') {
      left = panelRect.left - tooltipW - TOOLTIP_GAP;
      top = panelRect.top + panelRect.height / 2 - tooltipH / 2;
    } else if (pos === 'right') {
      left = panelRect.right + TOOLTIP_GAP;
      top = panelRect.top + panelRect.height / 2 - tooltipH / 2;
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

      <div className="tutorial-tooltip" style={getTooltipStyle()}>
        <div className={`tutorial-tooltip-arrow tutorial-tooltip-arrow--${step.tooltipPosition}`} />
        <div className="tutorial-tooltip-step">Step {stepIndex + 1} of {TUTORIAL_STEPS.length}</div>
        <div className="tutorial-tooltip-title">{step.title}</div>
        <div className="tutorial-tooltip-desc">{step.description}</div>
        <div className="tutorial-dots">
          {TUTORIAL_STEPS.map((_, i) => (
            <span
              key={i}
              className={`tutorial-dot ${i === stepIndex ? 'active' : ''} ${i < stepIndex ? 'completed' : ''}`}
            />
          ))}
        </div>
        <div className="tutorial-tooltip-hint">
          {stepIndex < TUTORIAL_STEPS.length - 1 ? 'Click anywhere to continue' : 'Click to finish'}
        </div>
      </div>
    </div>
  );
}
