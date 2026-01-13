/**
 * Native Helper Status Component
 *
 * Shows connection status in toolbar and opens a dialog for details/download.
 */

import { useState, useEffect, useCallback } from 'react';
import { NativeHelperClient, isNativeHelperAvailable } from '../../services/nativeHelper';
import type { SystemInfo, ConnectionStatus } from '../../services/nativeHelper';
import { useSettingsStore } from '../../stores/settingsStore';

// Direct download from app (bundled in public folder)
const HELPER_DIRECT_DOWNLOAD = '/downloads/masterselects-helper';

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
  useEffect(() => {
    checkConnection();

    // Subscribe to status changes
    const unsubscribe = NativeHelperClient.onStatusChange((newStatus) => {
      setStatus(newStatus);
      setNativeHelperConnected(newStatus === 'connected');
    });

    // Periodic check every 30 seconds (less aggressive)
    const interval = setInterval(checkConnection, 30000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [checkConnection, setNativeHelperConnected]);

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
  const [copied, setCopied] = useState(false);

  const { turboModeEnabled, setTurboModeEnabled } = useSettingsStore();

  const quickStartCommands = `chmod +x masterselects-helper && ./masterselects-helper`;

  // Fetch system info when connected
  useEffect(() => {
    if (status === 'connected') {
      NativeHelperClient.getInfo().then(setInfo).catch(() => setInfo(null));
    } else {
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

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(quickStartCommands);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

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
        <div className="welcome-tagline">
          <span className={isConnected ? 'welcome-tag-local' : 'welcome-tag-free'}>
            {isConnected ? '⚡ Connected' : '○ Not Running'}
          </span>
        </div>

        <h1 className="welcome-title" style={{ fontSize: '28px' }}>
          <span className="welcome-title-master">Native</span>
          <span className="welcome-title-selects">Helper</span>
        </h1>

        <p className="welcome-subtitle">
          10x faster ProRes & DNxHD decoding
        </p>

        {/* Content Card */}
        <div className="welcome-folder-card">
          <div className="info-content">
            {/* Enable Toggle */}
            <label className="flex items-center gap-3 mb-4 cursor-pointer">
              <input
                type="checkbox"
                checked={turboModeEnabled}
                onChange={(e) => setTurboModeEnabled(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span className="text-sm text-zinc-300">Enable Turbo Mode</span>
            </label>

            {isConnected && info ? (
              /* Connected State */
              <div className="space-y-3">
                <div className="info-features">
                  <div className="info-feature">
                    <span className="info-feature-icon">v{info.version}</span>
                    <span>Helper Version</span>
                  </div>
                  <div className="info-feature">
                    <span className="info-feature-icon">{info.cache_used_mb}MB</span>
                    <span>Cache Used ({info.cache_max_mb}MB max)</span>
                  </div>
                  {info.hw_accel.length > 0 && (
                    <div className="info-feature">
                      <span className="info-feature-icon">HW</span>
                      <span>{info.hw_accel.join(', ')}</span>
                    </div>
                  )}
                  <div className="info-feature">
                    <span className="info-feature-icon">{info.open_files}</span>
                    <span>Open Files</span>
                  </div>
                </div>

                <p className="text-xs text-green-400 text-center pt-2">
                  ProRes and DNxHD files will decode at native speed
                </p>
              </div>
            ) : turboModeEnabled ? (
              /* Not Connected State */
              <div className="space-y-4">
                <p className="text-sm text-zinc-400">
                  The Native Helper is a small companion app that provides hardware-accelerated
                  video decoding for professional codecs.
                </p>

                <a
                  href={HELPER_DIRECT_DOWNLOAD}
                  download="masterselects-helper"
                  className="block w-full text-center bg-blue-600 hover:bg-blue-500 text-white py-2.5 px-4 rounded-lg transition-colors font-medium"
                >
                  Download Helper (Linux, 1.8 MB)
                </a>

                <div className="bg-zinc-900 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs text-zinc-500">Quick start:</p>
                    <button
                      onClick={handleCopy}
                      className="text-xs text-zinc-500 hover:text-white transition-colors"
                    >
                      {copied ? '✓ Copied' : 'Copy'}
                    </button>
                  </div>
                  <code
                    className="text-xs text-zinc-300 font-mono block bg-zinc-800 p-2 rounded select-all cursor-text"
                    onClick={(e) => {
                      const selection = window.getSelection();
                      const range = document.createRange();
                      range.selectNodeContents(e.currentTarget);
                      selection?.removeAllRanges();
                      selection?.addRange(range);
                    }}
                  >
                    {quickStartCommands}
                  </code>
                </div>

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
              <p className="text-sm text-zinc-500 text-center py-4">
                Enable Turbo Mode to use hardware-accelerated decoding for professional codecs.
              </p>
            )}
          </div>
        </div>

        {/* Close Button */}
        <button className="welcome-enter" onClick={handleClose} style={{ marginTop: '16px' }}>
          <span>Close</span>
          <kbd>Esc</kbd>
        </button>
      </div>
    </div>
  );
}

export default NativeHelperStatus;
