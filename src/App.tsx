// WebVJ Mixer - Main Application

// DEV: Disable changelog dialog for development
const DEV_DISABLE_WHATS_NEW = false;

import { useState, useCallback, useEffect } from 'react';
import { Toolbar } from './components';
import { DockContainer } from './components/dock';
import { WelcomeOverlay } from './components/common/WelcomeOverlay';
import { WhatsNewDialog } from './components/common/WhatsNewDialog';
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

  if (testMode === 'parallel-decode') {
    return <ParallelDecodeTest />;
  }
  // Mobile detection
  const isMobile = useIsMobile();
  const forceMobile = useForceMobile();
  const forceDesktopMode = useSettingsStore((s) => s.forceDesktopMode);

  // Show mobile UI unless user explicitly requested desktop mode
  const showMobileUI = (isMobile || forceMobile) && !forceDesktopMode;

  // Render mobile UI if on mobile device (and not forcing desktop)
  if (showMobileUI) {
    return <MobileApp />;
  }
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

  // Load API keys from encrypted storage on mount
  const loadApiKeys = useSettingsStore((s) => s.loadApiKeys);
  useEffect(() => {
    loadApiKeys();
  }, [loadApiKeys]);

  // Check for stored project on mount, then poll for changes
  // This handles the case where Toolbar's restore fails and clears handles
  useEffect(() => {
    const checkProject = async () => {
      // Check both: IndexedDB handle exists AND project is actually open
      const hasHandle = await projectDB.hasLastProject();
      const isOpen = projectFileService.isProjectOpen();
      setHasStoredProject(hasHandle || isOpen);
      setIsChecking(false);
    };

    checkProject();

    // Poll for changes (handles cleared after failed restore)
    const interval = setInterval(async () => {
      const hasHandle = await projectDB.hasLastProject();
      const isOpen = projectFileService.isProjectOpen();
      setHasStoredProject(hasHandle || isOpen);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  // Show welcome if no stored project and not manually dismissed this session
  // Don't show while checking to avoid flash
  const showWelcome = !isChecking && !hasStoredProject && !manuallyDismissed;

  // Show What's New dialog after initial check (when no welcome overlay)
  useEffect(() => {
    if (DEV_DISABLE_WHATS_NEW) return;
    if (isChecking) return;

    // If welcome is showing, don't show What's New yet
    if (showWelcome) return;

    // Show What's New dialog
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
    </div>
  );
}

export default App;
