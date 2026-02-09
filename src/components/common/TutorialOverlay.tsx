import { useState, useEffect, useCallback, useRef } from 'react';
import { useDockStore } from '../../stores/dockStore';
import { useSettingsStore } from '../../stores/settingsStore';
import type { PanelType } from '../../types/dock';

const WELCOME_BUTTONS = [
  { id: 'premiere', label: 'Premiere Pro', logo: '/logo-premiere.svg' },
  { id: 'davinci', label: 'DaVinci Resolve', logo: '/logo-davinci.svg' },
  { id: 'finalcut', label: 'Final Cut Pro', logo: '/logo-finalcut.png' },
  { id: 'aftereffects', label: 'After Effects', logo: '/logo-aftereffects.svg' },
  { id: 'beginner', label: 'Beginner', logo: null },
] as const;

function ClippyMascot({ isClosing }: { isClosing: boolean }) {
  const [useWebP, setUseWebP] = useState(false);
  const [phase, setPhase] = useState<'intro' | 'loop' | 'outro'>('intro');
  const introRef = useRef<HTMLVideoElement>(null);
  const loopRef = useRef<HTMLVideoElement>(null);
  const outroRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const intro = introRef.current;
    if (!intro) return;
    const source = intro.querySelector('source');
    if (source) {
      source.addEventListener('error', () => setUseWebP(true), { once: true });
    }
    const onEnded = () => setPhase('loop');
    intro.addEventListener('ended', onEnded);
    return () => intro.removeEventListener('ended', onEnded);
  }, []);

  // Switch to outro when tutorial is closing
  useEffect(() => {
    if (isClosing && phase !== 'outro') {
      setPhase('outro');
      const outro = outroRef.current;
      if (outro) {
        outro.currentTime = 0;
        outro.play().catch(() => {});
      }
    }
  }, [isClosing, phase]);

  if (useWebP) {
    return <img src="/clippy.webp" alt="" className="tutorial-clippy" draggable={false} />;
  }

  return (
    <>
      <video
        ref={introRef}
        className="tutorial-clippy"
        autoPlay
        muted
        playsInline
        disablePictureInPicture
        style={{ display: phase === 'intro' ? undefined : 'none' }}
      >
        <source src="/clippy-intro.webm" type="video/webm" />
      </video>
      <video
        ref={loopRef}
        className="tutorial-clippy"
        autoPlay
        loop
        muted
        playsInline
        disablePictureInPicture
        style={{ display: phase === 'loop' ? undefined : 'none' }}
      >
        <source src="/clippy.webm" type="video/webm" />
      </video>
      <video
        ref={outroRef}
        className="tutorial-clippy"
        preload="auto"
        muted
        playsInline
        disablePictureInPicture
        style={{ display: phase === 'outro' ? undefined : 'none' }}
      >
        <source src="/clippy-outro.webm" type="video/webm" />
      </video>
    </>
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
  // stepIndex -1 = welcome screen (only part 1), 0+ = normal steps
  const [stepIndex, setStepIndex] = useState(part === 1 ? -1 : 0);
  const [panelRect, setPanelRect] = useState<DOMRect | null>(null);
  const [highlightRect, setHighlightRect] = useState<DOMRect | null>(null);
  const [isClosing, setIsClosing] = useState(false);
  const activatePanelType = useDockStore((s) => s.activatePanelType);
  const setUserBackground = useSettingsStore((s) => s.setUserBackground);
  const closingRef = useRef(false);

  const isPart2 = part === 2;
  const isWelcome = stepIndex === -1;
  const steps = isPart2 ? TIMELINE_STEPS : PANEL_STEPS;
  const step = isWelcome ? null : steps[stepIndex];

  // Find and measure targets
  const measureTargets = useCallback(() => {
    if (isWelcome) {
      // No panel to highlight during welcome
      setPanelRect(null);
      setHighlightRect(null);
      return;
    }
    if (isPart2) {
      const timelineEl = document.querySelector('[data-group-id="timeline-group"]');
      if (timelineEl) {
        setPanelRect(timelineEl.getBoundingClientRect());
      } else {
        setPanelRect(null);
      }
      const targetEl = document.querySelector((step as TimelineStep).selector);
      if (targetEl) {
        setHighlightRect(targetEl.getBoundingClientRect());
      } else {
        setHighlightRect(null);
      }
    } else {
      const el = document.querySelector(`[data-group-id="${(step as PanelStep).groupId}"]`);
      if (el) {
        setPanelRect(el.getBoundingClientRect());
      } else {
        setPanelRect(null);
      }
      setHighlightRect(null);
    }
  }, [isPart2, isWelcome, step]);

  // Activate the correct tab and measure on step change
  useEffect(() => {
    if (!isPart2 && step) {
      activatePanelType((step as PanelStep).panelType);
    }
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
    setTimeout(onClose, 1800);
  }, [onClose]);

  const advance = useCallback(() => {
    if (isClosing || isWelcome) return;
    if (stepIndex < steps.length - 1) {
      setStepIndex(stepIndex + 1);
    } else {
      close();
    }
  }, [stepIndex, steps.length, isClosing, isWelcome, close]);

  // Escape to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [close]);

  // Compute tooltip position: centered for welcome, anchored to panel for steps
  const getTooltipStyle = (): React.CSSProperties => {
    if (isWelcome) {
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const tooltipW = 480;
      const tooltipH = 220;
      return {
        left: vw / 2 - tooltipW / 2,
        top: vh / 2 - tooltipH / 2,
      };
    }
    const anchorRect = isPart2 ? highlightRect : panelRect;
    if (!anchorRect) return { opacity: 0 };

    const tooltipW = 380;
    const tooltipH = 180;
    const pos = step!.tooltipPosition;
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

    left = Math.max(12, Math.min(left, vw - tooltipW - 12));
    top = Math.max(12, Math.min(top, vh - tooltipH - 12));

    return { left, top };
  };

  const handleWelcomeSelect = useCallback((id: string) => {
    setUserBackground(id);
    setStepIndex(0);
  }, [setUserBackground]);

  const tooltipStyle = getTooltipStyle();

  return (
    <div
      className={`tutorial-backdrop ${isClosing ? 'closing' : ''}`}
      onClick={isWelcome ? undefined : advance}
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
          mask={panelRect ? 'url(#tutorial-mask)' : undefined}
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

      {/* Clippy in own container so it doesn't fade with tooltip */}
      <div className={`tutorial-clippy-wrapper ${isWelcome ? 'tutorial-clippy-wrapper--welcome' : ''}`} style={tooltipStyle}>
        <ClippyMascot isClosing={isClosing} />
      </div>

      <div
        className={`tutorial-tooltip ${isWelcome ? 'tutorial-tooltip--welcome' : ''}`}
        style={tooltipStyle}
      >
        {step && (
          <div className={`tutorial-tooltip-arrow tutorial-tooltip-arrow--${step.tooltipPosition}`} />
        )}
        <div className="tutorial-tooltip-content">
          <div className="tutorial-tooltip-text">
            {isWelcome ? (
              <>
                <div className="tutorial-welcome-title">Welcome! Where are you coming from?</div>
                <div className="tutorial-welcome-subtitle">This helps us tailor tips to your experience</div>
                <div className="tutorial-welcome-grid">
                  {WELCOME_BUTTONS.map((btn) => (
                    <button
                      key={btn.id}
                      className="tutorial-welcome-btn"
                      onClick={() => handleWelcomeSelect(btn.id)}
                    >
                      <div className="tutorial-welcome-icon">
                        {btn.logo ? (
                          <img src={btn.logo} alt={btn.label} draggable={false} />
                        ) : (
                          <span className="tutorial-welcome-icon--beginner">★</span>
                        )}
                      </div>
                      <div className="tutorial-welcome-label">{btn.label}</div>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="tutorial-tooltip-step">Step {stepIndex + 1} of {steps.length}</div>
                <div className="tutorial-tooltip-title">{step!.title}</div>
                <div className="tutorial-tooltip-desc">{step!.description}</div>
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
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
