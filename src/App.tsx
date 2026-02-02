// WebVJ Mixer - Main Application

// DEV: Disable changelog dialog for development
const DEV_DISABLE_WHATS_NEW = false;

import { useState, useCallback, useEffect } from 'react';
import { Toolbar } from './components';
import { DockContainer } from './components/dock';
import { WelcomeOverlay } from './components/common/WelcomeOverlay';
import { WhatsNewDialog } from './components/common/WhatsNewDialog';
import { IndexedDBErrorDialog } from './components/common/IndexedDBErrorDialog';
import { MobileApp } from './components/mobile';
import { useGlobalHistory } from './hooks/useGlobalHistory';
import { useClipPanelSync } from './hooks/useClipPanelSync';
import { useIsMobile, useForceMobile } from './hooks/useIsMobile';
import { useSettingsStore } from './stores/settingsStore';
import { projectDB } from './services/projectDB';
import { projectFileService } from './services/projectFileService';
import './App.css';

// Dev test pages - access via ?test=parallel-decode
import { ParallelDecodeTest } from './test/ParallelDecodeTest';

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
    }, 500);

    return () => clearInterval(interval);
  }, []);

  // Show welcome if no stored project and not manually dismissed this session
  // Don't show while checking to avoid flash
  const showWelcome = !isChecking && !hasStoredProject && !manuallyDismissed;

  // Show What's New dialog after initial check (when no welcome overlay)
  // This effect intentionally sets state based on derived conditions
  useEffect(() => {
    if (DEV_DISABLE_WHATS_NEW) return;
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
    if (!DEV_DISABLE_WHATS_NEW) {
      setTimeout(() => setShowWhatsNew(true), 300);
    }
  }, []);

  const handleWhatsNewClose = useCallback(() => {
    setShowWhatsNew(false);
  }, []);

  const handleIndexedDBErrorClose = useCallback(() => {
    setShowIndexedDBError(false);
  }, []);

  // === EARLY RETURNS AFTER ALL HOOKS ===

  // Test mode
  if (testMode === 'parallel-decode') {
    return <ParallelDecodeTest />;
  }

  // Show mobile UI unless user explicitly requested desktop mode
  const showMobileUI = (isMobile || forceMobile) && !forceDesktopMode;
  if (showMobileUI) {
    return <MobileApp />;
  }

  return (
    <div className="app">
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
    </div>
  );
}

export default App;
