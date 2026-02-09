// WebVJ Mixer - Main Application

// Changelog visibility controlled by Vite define:
// npm run dev          → hidden (default)
// npm run dev:changelog → shown
// npm run build        → always shown
declare const __SHOW_CHANGELOG__: boolean;
const SHOW_CHANGELOG = typeof __SHOW_CHANGELOG__ !== 'undefined' ? __SHOW_CHANGELOG__ : true;

import { useState, useCallback, useEffect, lazy, Suspense } from 'react';
import { Toolbar } from './components';
import { DockContainer } from './components/dock';
import { WelcomeOverlay } from './components/common/WelcomeOverlay';
import { WhatsNewDialog } from './components/common/WhatsNewDialog';
import { IndexedDBErrorDialog } from './components/common/IndexedDBErrorDialog';
import { LinuxVulkanWarning } from './components/common/LinuxVulkanWarning';
import { TutorialOverlay } from './components/common/TutorialOverlay';
import { MobileApp } from './components/mobile';
import { useGlobalHistory } from './hooks/useGlobalHistory';
import { useClipPanelSync } from './hooks/useClipPanelSync';
import { useIsMobile, useForceMobile } from './hooks/useIsMobile';
import { useSettingsStore } from './stores/settingsStore';
import { projectDB } from './services/projectDB';
import { projectFileService } from './services/projectFileService';
import './App.css';

// Dev test pages - lazy loaded to avoid bloating main bundle
// Access via ?test=parallel-decode
const ParallelDecodeTest = lazy(() =>
  import('./test/ParallelDecodeTest').then(m => ({ default: m.ParallelDecodeTest }))
);

function App() {
  // Check for test mode via URL param
  const urlParams = new URLSearchParams(window.location.search);
  const testMode = urlParams.get('test');

  // === ALL HOOKS MUST BE CALLED BEFORE ANY EARLY RETURNS ===

  // Mobile detection
  const isMobile = useIsMobile();
  const forceMobile = useForceMobile();
  const forceDesktopMode = useSettingsStore((s) => s.forceDesktopMode);

  // Initialize global undo/redo system
  useGlobalHistory();

  // Auto-switch panels based on clip selection
  useClipPanelSync();

  // Check if there's a stored project in IndexedDB (the only allowed browser storage)
  const [isChecking, setIsChecking] = useState(true);
  const [hasStoredProject, setHasStoredProject] = useState(false);
  const [manuallyDismissed, setManuallyDismissed] = useState(false);

  // What's New dialog state - show on every refresh after welcome (if any)
  const [showWhatsNew, setShowWhatsNew] = useState(false);

  // Tutorial state
  const [showTutorial, setShowTutorial] = useState(false);
  const [tutorialPart, setTutorialPart] = useState<1 | 2>(1);
  const hasSeenTutorial = useSettingsStore((s) => s.hasSeenTutorial);
  const setHasSeenTutorial = useSettingsStore((s) => s.setHasSeenTutorial);
  const hasSeenTutorialPart2 = useSettingsStore((s) => s.hasSeenTutorialPart2);
  const setHasSeenTutorialPart2 = useSettingsStore((s) => s.setHasSeenTutorialPart2);

  // IndexedDB error dialog state
  const [showIndexedDBError, setShowIndexedDBError] = useState(false);

  // Load API keys from encrypted storage on mount
  const loadApiKeys = useSettingsStore((s) => s.loadApiKeys);
  useEffect(() => {
    loadApiKeys();
  }, [loadApiKeys]);

  // Check for stored project on mount, then poll for changes
  // This handles the case where Toolbar's restore fails and clears handles
  useEffect(() => {
    const checkProject = async () => {
      // Check if IndexedDB has failed to initialize
      if (projectDB.hasInitFailed()) {
        setShowIndexedDBError(true);
        setIsChecking(false);
        return;
      }

      try {
        // Check both: IndexedDB handle exists AND project is actually open
        const hasHandle = await projectDB.hasLastProject();
        const isOpen = projectFileService.isProjectOpen();
        setHasStoredProject(hasHandle || isOpen);
      } catch {
        // If hasLastProject fails, IndexedDB is corrupted
        if (projectDB.hasInitFailed()) {
          setShowIndexedDBError(true);
        }
      }
      setIsChecking(false);
    };

    checkProject();

    // Poll for changes (handles cleared after failed restore)
    // Using 2000ms interval to reduce CPU usage - project state changes are rare
    const interval = setInterval(async () => {
      // Check if IndexedDB has failed (could happen after initial load)
      if (projectDB.hasInitFailed()) {
        setShowIndexedDBError(true);
        return;
      }

      try {
        const hasHandle = await projectDB.hasLastProject();
        const isOpen = projectFileService.isProjectOpen();
        setHasStoredProject(hasHandle || isOpen);
      } catch {
        if (projectDB.hasInitFailed()) {
          setShowIndexedDBError(true);
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Show welcome if no stored project and not manually dismissed this session
  // Don't show while checking to avoid flash
  const showWelcome = !isChecking && !hasStoredProject && !manuallyDismissed;

  // Show What's New dialog after initial check (when no welcome overlay)
  // This effect intentionally sets state based on derived conditions
  useEffect(() => {
    if (!SHOW_CHANGELOG) return;
    if (isChecking) return;

    // If welcome is showing, don't show What's New yet
    if (showWelcome) return;

    // Show What's New dialog - this is intentional state sync, not a cascading render
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setShowWhatsNew(true);
  }, [isChecking, showWelcome]);

  const handleWelcomeComplete = useCallback(() => {
    setManuallyDismissed(true);
    setHasStoredProject(true); // Project was just created
    // After welcome, show What's New with small delay for animation
    if (!!SHOW_CHANGELOG) {
      setTimeout(() => setShowWhatsNew(true), 300);
    } else if (!hasSeenTutorial) {
      // No changelog → start tutorial directly
      setTimeout(() => setShowTutorial(true), 200);
    }
  }, [hasSeenTutorial]);

  const handleWhatsNewClose = useCallback(() => {
    setShowWhatsNew(false);
    if (!hasSeenTutorial) {
      setTimeout(() => setShowTutorial(true), 200);
    }
  }, [hasSeenTutorial]);

  const handleTutorialClose = useCallback(() => {
    if (tutorialPart === 1 && !hasSeenTutorialPart2) {
      // Part 1 finished, auto-start Part 2
      setTutorialPart(2);
      setHasSeenTutorial(true);
    } else if (tutorialPart === 2) {
      // Part 2 finished
      setShowTutorial(false);
      setHasSeenTutorial(true);
      setHasSeenTutorialPart2(true);
    } else {
      // Part 1 re-triggered manually (Part 2 already seen)
      setShowTutorial(false);
    }
  }, [tutorialPart, hasSeenTutorialPart2, setHasSeenTutorial, setHasSeenTutorialPart2]);

  // Listen for manual tutorial trigger from View menu
  useEffect(() => {
    const handleStartTutorial = () => {
      setTutorialPart(1);
      setShowTutorial(true);
    };
    const handleStartTimelineTutorial = () => {
      setTutorialPart(2);
      setShowTutorial(true);
    };
    window.addEventListener('start-tutorial', handleStartTutorial);
    window.addEventListener('start-timeline-tutorial', handleStartTimelineTutorial);
    return () => {
      window.removeEventListener('start-tutorial', handleStartTutorial);
      window.removeEventListener('start-timeline-tutorial', handleStartTimelineTutorial);
    };
  }, []);

  const handleIndexedDBErrorClose = useCallback(() => {
    setShowIndexedDBError(false);
  }, []);

  // === EARLY RETURNS AFTER ALL HOOKS ===

  // Test mode - wrapped in Suspense for lazy-loaded component
  if (testMode === 'parallel-decode') {
    return (
      <Suspense fallback={<div style={{ padding: 20 }}>Loading test...</div>}>
        <ParallelDecodeTest />
      </Suspense>
    );
  }

  // Show mobile UI unless user explicitly requested desktop mode
  const showMobileUI = (isMobile || forceMobile) && !forceDesktopMode;
  if (showMobileUI) {
    return <MobileApp />;
  }

  return (
    <div className="app">
      <LinuxVulkanWarning />
      <Toolbar />
      <DockContainer />
      {showWelcome && (
        <WelcomeOverlay onComplete={handleWelcomeComplete} noFadeOnClose />
      )}
      {showWhatsNew && (
        <WhatsNewDialog onClose={handleWhatsNewClose} />
      )}
      {showIndexedDBError && (
        <IndexedDBErrorDialog onClose={handleIndexedDBErrorClose} />
      )}
      {showTutorial && (
        <TutorialOverlay key={tutorialPart} onClose={handleTutorialClose} part={tutorialPart} />
      )}
    </div>
  );
}

export default App;
