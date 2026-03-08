/**
 * Native Helper Status Component
 *
 * Shows connection status in toolbar and opens a dialog for details/download.
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { NativeHelperClient, isNativeHelperAvailable } from '../../services/nativeHelper';
import type { SystemInfo, ConnectionStatus } from '../../services/nativeHelper';
import { useSettingsStore } from '../../stores/settingsStore';
import { upgradeAllClipsToNativeDecoder, downgradeAllClipsFromNativeDecoder, startClipWatcher, stopClipWatcher } from '../../stores/timeline/clip/upgradeToNativeDecoder';

// Detect platform
function detectPlatform(): 'mac' | 'windows' | 'linux' | 'unknown' {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes('mac')) return 'mac';
  if (ua.includes('win')) return 'windows';
  if (ua.includes('linux')) return 'linux';
  return 'unknown';
}

// GitHub releases download URLs
const NATIVE_HELPER_RELEASE = 'https://github.com/Sportinger/MasterSelects/releases/tag/native-helper-v0.2.0';
const GITHUB_RELEASES = NATIVE_HELPER_RELEASE;
const DOWNLOAD_LINKS = {
  windows: 'https://github.com/Sportinger/MasterSelects/releases/download/native-helper-v0.2.0/MasterSelects-NativeHelper-v0.2.0-windows-x64.msi',
  mac: NATIVE_HELPER_RELEASE,
  linux: NATIVE_HELPER_RELEASE,
} as const;



/**
 * Toolbar button that shows helper status
 */
export function NativeHelperStatus() {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [showDialog, setShowDialog] = useState(false);

  const { turboModeEnabled, setNativeHelperConnected } = useSettingsStore();

  // Check connection status
  const checkConnection = useCallback(async () => {
    if (!turboModeEnabled) {
      setStatus('disconnected');
      setNativeHelperConnected(false);
      return;
    }

    try {
      const available = await isNativeHelperAvailable();
      setStatus(available ? 'connected' : 'disconnected');
      setNativeHelperConnected(available);
    } catch {
      setStatus('disconnected');
      setNativeHelperConnected(false);
    }
  }, [turboModeEnabled, setNativeHelperConnected]);

  // Check on mount and when turbo mode changes
  // This effect syncs with external NativeHelper service state
  useEffect(() => {
    // Use void to explicitly mark fire-and-forget async call
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void checkConnection();

    // Subscribe to status changes
    const unsubscribe = NativeHelperClient.onStatusChange((newStatus) => {
      setStatus(newStatus);
      setNativeHelperConnected(newStatus === 'connected');
    });

    // Periodic check every 30 seconds (less aggressive)
    const interval = setInterval(() => void checkConnection(), 30000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [checkConnection, setNativeHelperConnected]);

  // Upgrade/downgrade clips when native decode setting or connection changes
  const nativeDecodeEnabled = useSettingsStore((s) => s.nativeDecodeEnabled);
  const prevConnectedRef = useRef(false);
  useEffect(() => {
    const isNowConnected = status === 'connected' && nativeDecodeEnabled;
    const wasConnected = prevConnectedRef.current;
    prevConnectedRef.current = isNowConnected;

    if (isNowConnected && !wasConnected) {
      // Helper connected + decode enabled — upgrade all clips + watch for new ones
      void upgradeAllClipsToNativeDecoder();
      startClipWatcher();
    } else if (!isNowConnected && wasConnected) {
      // Helper disconnected or decode off — downgrade + stop watching
      stopClipWatcher();
      downgradeAllClipsFromNativeDecoder();
    }
  }, [status, nativeDecodeEnabled]);

  const isConnected = status === 'connected';

  return (
    <>
      <button
        onClick={() => setShowDialog(true)}
        className="p-1 rounded hover:bg-white/10 transition-colors"
        title={isConnected ? 'Turbo Mode active' : 'Turbo Mode'}
        style={{ background: 'transparent', lineHeight: 1 }}
      >
        {isConnected ? (
          <span style={{ fontSize: '14px', color: '#facc15' }}>⚡</span>
        ) : (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#52525b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
        )}
      </button>

      {showDialog && (
        <NativeHelperDialog
          status={status}
          onClose={() => setShowDialog(false)}
          onRetry={checkConnection}
        />
      )}
    </>
  );
}

/**
 * Modal dialog for Native Helper details
 */
function NativeHelperDialog({
  status,
  onClose,
  onRetry,
}: {
  status: ConnectionStatus;
  onClose: () => void;
  onRetry: () => void;
}) {
  const [isClosing, setIsClosing] = useState(false);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [checking, setChecking] = useState(false);

  const { turboModeEnabled, setTurboModeEnabled, nativeDecodeEnabled, setNativeDecodeEnabled } = useSettingsStore();

  const platform = useMemo(() => detectPlatform(), []);

  // Fetch system info when connected
  // This effect syncs UI state with external NativeHelper info
  useEffect(() => {
    if (status === 'connected') {
      NativeHelperClient.getInfo().then(setInfo).catch(() => setInfo(null));
    } else {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setInfo(null);
    }
  }, [status]);

  const handleClose = useCallback(() => {
    if (isClosing) return;
    setIsClosing(true);
    setTimeout(onClose, 150);
  }, [onClose, isClosing]);

  const handleRetry = async () => {
    setChecking(true);
    await onRetry();
    setChecking(false);
  };

  // Auto-check connection when dialog opens
  useEffect(() => {
    if (turboModeEnabled && status !== 'connected') {
      void handleRetry();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleClose]);

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) handleClose();
  };

  const isConnected = status === 'connected';
  const downloadLink = (platform !== 'unknown' && DOWNLOAD_LINKS[platform]) || GITHUB_RELEASES;
  const platformLabel = platform === 'mac' ? 'macOS' : platform === 'windows' ? 'Windows' : platform === 'linux' ? 'Linux' : 'your platform';

  return (
    <div
      className="welcome-overlay-backdrop"
      onClick={handleBackdropClick}
      style={{
        animation: 'none',
        opacity: isClosing ? 0 : 1,
        transition: 'opacity 150ms ease-out',
      }}
    >
      <div
        className="welcome-overlay"
        style={{
          maxWidth: '480px',
          animation: 'none',
          opacity: isClosing ? 0 : 1,
          transform: isClosing ? 'scale(0.95)' : 'none',
          transition: 'opacity 150ms ease-out, transform 150ms ease-out',
        }}
      >
        {/* Header */}
        <div className="welcome-tagline" style={{ animation: 'welcome-fade-in 0.3s ease-out both' }}>
          <span style={{ color: isConnected ? '#51cf66' : '#ff6b6b' }}>
            {isConnected ? '⚡ Connected' : '○ Not Running'}
          </span>
        </div>

        <h1 className="welcome-title" style={{ fontSize: '28px' }}>
          <span className="welcome-title-master" style={{ animation: 'welcome-title-master-in 0.4s ease-out both' }}>Native</span>
          <span className="welcome-title-selects" style={{ animation: 'welcome-title-selects-in 0.4s ease-out 0.1s both' }}>Helper</span>
        </h1>

        <p className="welcome-subtitle" style={{ animation: 'welcome-fade-up 0.4s ease-out 0.15s both' }}>
          Downloads, project saving & more
        </p>

        {/* Content Card */}
        <div className="welcome-folder-card" style={{ animation: 'welcome-fade-up 0.4s ease-out 0.2s both' }}>
          <div className="info-content">
            {/* Enable Toggles */}
            <label className="flex items-center gap-3 mb-2 cursor-pointer">
              <input
                type="checkbox"
                checked={turboModeEnabled}
                onChange={(e) => setTurboModeEnabled(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span className="text-sm text-zinc-300">Enable Native Helper (Downloads)</span>
            </label>
            <label className="flex items-center gap-3 mb-4 cursor-pointer">
              <input
                type="checkbox"
                checked={nativeDecodeEnabled}
                onChange={(e) => setNativeDecodeEnabled(e.target.checked)}
                className="w-4 h-4 rounded"
                disabled={!turboModeEnabled}
              />
              <span className={`text-sm ${turboModeEnabled ? 'text-zinc-300' : 'text-zinc-500'}`}>Turbo Decode/Encode (FFmpeg)</span>
            </label>

            {isConnected && info ? (
              /* Connected State */
              <div className="space-y-3">
                <div className="info-features">
                  <div className="info-feature">
                    <span className="info-feature-icon">v{info.version}</span>
                    <span>Helper Version</span>
                  </div>
                  {info.cache_used_mb !== undefined && (
                    <div className="info-feature">
                      <span className="info-feature-icon">{info.cache_used_mb}MB</span>
                      <span>Cache Used ({info.cache_max_mb}MB max)</span>
                    </div>
                  )}
                  {info.hw_accel && info.hw_accel.length > 0 && (
                    <div className="info-feature">
                      <span className="info-feature-icon">HW</span>
                      <span>{info.hw_accel.join(', ')}</span>
                    </div>
                  )}
                  {info.open_files !== undefined && (
                    <div className="info-feature">
                      <span className="info-feature-icon">{info.open_files}</span>
                      <span>Open Files</span>
                    </div>
                  )}
                  <div className="info-feature">
                    <span className="info-feature-icon">{(info as any).ytdlp_available ? '✓' : '✗'}</span>
                    <span>Video Downloads</span>
                  </div>
                  <div className="info-feature">
                    <span className="info-feature-icon">{(info as any).fs_commands ? '✓' : '✗'}</span>
                    <span>Project Saving</span>
                  </div>
                </div>

                <p className="text-xs text-green-400 text-center pt-2">
                  {[
                    (info as any).ytdlp_available && 'Downloads',
                    !(info as any).lite && 'FFmpeg',
                    (info as any).fs_commands && 'Projects',
                  ].filter(Boolean).join(' · ') || 'Connected'}
                </p>

                {/* Download link always visible */}
                <a
                  href={downloadLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full text-center text-xs text-zinc-500 hover:text-zinc-300 py-1 transition-colors"
                >
                  Download latest version for {platformLabel}
                </a>
              </div>
            ) : turboModeEnabled ? (
              /* Not Connected State */
              <div className="space-y-4">
                <p className="text-sm text-zinc-400">
                  Download the Native Helper to enable video downloads, project saving in Firefox, and FFmpeg encoding.
                </p>

                <a
                  href={downloadLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full text-center bg-blue-600 hover:bg-blue-500 text-white py-2.5 px-4 rounded-lg transition-colors font-medium"
                >
                  Download for {platformLabel}
                </a>

                {platform === 'windows' && (
                  <div className="bg-zinc-900 rounded-lg p-3 space-y-2">
                    <p className="text-xs text-zinc-500 font-medium">Installation:</p>
                    <ol className="text-xs text-zinc-400 space-y-1.5 list-decimal list-inside">
                      <li>Run the MSI installer</li>
                      <li>The helper starts in the system tray (notification area)</li>
                      <li>Right-click the tray icon for options</li>
                    </ol>
                  </div>
                )}

                {platform === 'mac' && (
                  <div className="bg-zinc-900 rounded-lg p-3 space-y-2">
                    <p className="text-xs text-zinc-500 font-medium">Installation:</p>
                    <ol className="text-xs text-zinc-400 space-y-1.5 list-decimal list-inside">
                      <li>Open the DMG and drag the app to Applications</li>
                      <li>Open from Applications (right-click → Open for first launch)</li>
                      <li>The helper runs in your menubar</li>
                    </ol>
                  </div>
                )}

                {platform === 'mac' && (
                  <div className="bg-zinc-900 rounded-lg p-3">
                    <p className="text-xs text-zinc-500 mb-2">For YouTube downloads, also install yt-dlp:</p>
                    <code
                      className="text-xs text-zinc-300 font-mono block bg-zinc-800 p-2 rounded select-all cursor-pointer"
                      onClick={(e) => {
                        navigator.clipboard.writeText('brew install yt-dlp');
                        const el = e.currentTarget;
                        el.textContent = '✓ Copied!';
                        setTimeout(() => { el.textContent = 'brew install yt-dlp'; }, 1500);
                      }}
                    >
                      brew install yt-dlp
                    </code>
                  </div>
                )}

                {platform === 'linux' && (
                  <div className="bg-zinc-900 rounded-lg p-3">
                    <p className="text-xs text-zinc-500 mb-2">Install yt-dlp for downloads:</p>
                    <code
                      className="text-xs text-zinc-300 font-mono block bg-zinc-800 p-2 rounded select-all cursor-pointer"
                      onClick={(e) => {
                        navigator.clipboard.writeText('sudo apt install yt-dlp || pip install yt-dlp');
                        const el = e.currentTarget;
                        el.textContent = '✓ Copied!';
                        setTimeout(() => { el.textContent = 'sudo apt install yt-dlp || pip install yt-dlp'; }, 1500);
                      }}
                    >
                      sudo apt install yt-dlp || pip install yt-dlp
                    </code>
                  </div>
                )}

                <button
                  onClick={handleRetry}
                  disabled={checking}
                  className="w-full text-center text-sm text-zinc-400 hover:text-white py-2 transition-colors disabled:opacity-50"
                >
                  {checking ? 'Checking...' : 'Check Connection'}
                </button>
              </div>
            ) : (
              /* Disabled State */
              <div className="space-y-4">
                <p className="text-sm text-zinc-500 text-center py-2">
                  Enable Native Helper above to use video downloads, project saving and FFmpeg decoding.
                </p>
                <a
                  href={downloadLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full text-center text-xs text-zinc-500 hover:text-zinc-300 py-1 transition-colors"
                >
                  Download for {platformLabel}
                </a>
              </div>
            )}
          </div>
        </div>

        {/* Close Button */}
        <button className="welcome-enter" onClick={handleClose} style={{ animation: 'welcome-fade-up 0.3s ease-out 0.25s both', marginTop: '16px' }}>
          <span>Close</span>
          <kbd>Esc</kbd>
        </button>
      </div>
    </div>
  );
}

export default NativeHelperStatus;
