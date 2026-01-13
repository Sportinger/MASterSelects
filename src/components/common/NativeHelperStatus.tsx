/**
 * Native Helper Status Component
 *
 * Shows the connection status of the native helper and provides
 * download links when not connected.
 */

import { useState, useEffect, useCallback } from 'react';
import { NativeHelperClient, isNativeHelperAvailable } from '../../services/nativeHelper';
import type { SystemInfo, ConnectionStatus } from '../../services/nativeHelper';
import { useSettingsStore } from '../../stores/settingsStore';

// GitHub releases URL for the helper
const HELPER_DOWNLOAD_URL = 'https://github.com/Sportinger/MASterSelects/releases/latest';
const HELPER_DIRECT_DOWNLOAD = 'https://github.com/Sportinger/MASterSelects/releases/latest/download/masterselects-helper';

interface NativeHelperStatusProps {
  /** Show as compact icon only */
  compact?: boolean;
  /** Show in toolbar style */
  toolbar?: boolean;
}

export function NativeHelperStatus({ compact = false, toolbar = false }: NativeHelperStatusProps) {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [checking, setChecking] = useState(true);
  const [showPopover, setShowPopover] = useState(false);

  const { turboModeEnabled, setNativeHelperConnected } = useSettingsStore();

  // Check connection status
  const checkConnection = useCallback(async () => {
    if (!turboModeEnabled) {
      setStatus('disconnected');
      setInfo(null);
      setNativeHelperConnected(false);
      setChecking(false);
      return;
    }

    setChecking(true);
    try {
      const available = await isNativeHelperAvailable();
      if (available) {
        setStatus('connected');
        setNativeHelperConnected(true);
        // Get system info
        try {
          const systemInfo = await NativeHelperClient.getInfo();
          setInfo(systemInfo);
        } catch {
          // Info fetch failed but still connected
        }
      } else {
        setStatus('disconnected');
        setNativeHelperConnected(false);
        setInfo(null);
      }
    } catch {
      setStatus('error');
      setNativeHelperConnected(false);
      setInfo(null);
    }
    setChecking(false);
  }, [turboModeEnabled, setNativeHelperConnected]);

  // Check on mount and when turbo mode changes
  useEffect(() => {
    checkConnection();

    // Subscribe to status changes
    const unsubscribe = NativeHelperClient.onStatusChange((newStatus) => {
      setStatus(newStatus);
      setNativeHelperConnected(newStatus === 'connected');
      if (newStatus === 'connected') {
        NativeHelperClient.getInfo().then(setInfo).catch(() => {});
      } else {
        setInfo(null);
      }
    });

    // Periodic check every 10 seconds
    const interval = setInterval(checkConnection, 10000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [checkConnection, setNativeHelperConnected]);

  // Status icon and color
  const getStatusIcon = () => {
    if (checking) return '...';
    switch (status) {
      case 'connected':
        return '⚡';
      case 'connecting':
        return '...';
      case 'error':
        return '⚠';
      default:
        return '○';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return '#4ade80'; // green
      case 'connecting':
        return '#fbbf24'; // yellow
      case 'error':
        return '#f87171'; // red
      default:
        return '#6b7280'; // gray
    }
  };

  const getStatusText = () => {
    if (checking) return 'Checking...';
    switch (status) {
      case 'connected':
        return 'Turbo Mode Active';
      case 'connecting':
        return 'Connecting...';
      case 'error':
        return 'Connection Error';
      default:
        return 'Turbo Mode Available';
    }
  };

  // Compact toolbar icon
  if (toolbar) {
    return (
      <div className="relative">
        <button
          onClick={() => setShowPopover(!showPopover)}
          className="flex items-center gap-1 px-2 py-1 rounded text-xs hover:bg-white/10 transition-colors"
          title={getStatusText()}
        >
          <span style={{ color: getStatusColor() }}>{getStatusIcon()}</span>
          {status === 'connected' && (
            <span className="text-green-400 font-medium">Turbo</span>
          )}
        </button>

        {showPopover && (
          <div
            className="absolute top-full right-0 mt-1 w-72 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-50 p-3"
            onMouseLeave={() => setShowPopover(false)}
          >
            <NativeHelperPopoverContent
              status={status}
              info={info}
              checking={checking}
              onRetry={checkConnection}
            />
          </div>
        )}
      </div>
    );
  }

  // Compact icon only
  if (compact) {
    return (
      <span
        style={{ color: getStatusColor() }}
        title={getStatusText()}
        className="cursor-help"
      >
        {getStatusIcon()}
      </span>
    );
  }

  // Full status display
  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
      <NativeHelperPopoverContent
        status={status}
        info={info}
        checking={checking}
        onRetry={checkConnection}
      />
    </div>
  );
}

// Shared popover content
function NativeHelperPopoverContent({
  status,
  info,
  checking,
  onRetry,
}: {
  status: ConnectionStatus;
  info: SystemInfo | null;
  checking: boolean;
  onRetry: () => void;
}) {
  const { turboModeEnabled, setTurboModeEnabled } = useSettingsStore();

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-white flex items-center gap-2">
          <span>Native Helper</span>
          {status === 'connected' && (
            <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
              Connected
            </span>
          )}
        </h3>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={turboModeEnabled}
            onChange={(e) => setTurboModeEnabled(e.target.checked)}
            className="rounded"
          />
          <span className="text-zinc-400">Enable</span>
        </label>
      </div>

      {/* Connected state */}
      {status === 'connected' && info && (
        <div className="text-xs space-y-1.5 text-zinc-400">
          <div className="flex justify-between">
            <span>Version:</span>
            <span className="text-zinc-300">v{info.version}</span>
          </div>
          <div className="flex justify-between">
            <span>Cache:</span>
            <span className="text-zinc-300">
              {info.cache_used_mb} / {info.cache_max_mb} MB
            </span>
          </div>
          {info.hw_accel.length > 0 && (
            <div className="flex justify-between">
              <span>HW Accel:</span>
              <span className="text-zinc-300">{info.hw_accel.join(', ')}</span>
            </div>
          )}
          <div className="flex justify-between">
            <span>Open files:</span>
            <span className="text-zinc-300">{info.open_files}</span>
          </div>
          <div className="pt-2 text-green-400">
            ProRes & DNxHD decoding at native speed
          </div>
        </div>
      )}

      {/* Disconnected state */}
      {status !== 'connected' && turboModeEnabled && (
        <div className="space-y-3">
          <p className="text-xs text-zinc-400">
            Native Helper provides 10x faster ProRes/DNxHD decoding with hardware acceleration.
          </p>

          <div className="space-y-2">
            <a
              href={HELPER_DIRECT_DOWNLOAD}
              download="masterselects-helper"
              className="block w-full text-center bg-blue-600 hover:bg-blue-500 text-white text-sm py-2 px-3 rounded transition-colors"
            >
              Download Helper (Linux)
            </a>

            <a
              href={HELPER_DOWNLOAD_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="block w-full text-center bg-zinc-700 hover:bg-zinc-600 text-zinc-300 text-xs py-1.5 px-3 rounded transition-colors"
            >
              All Releases
            </a>
          </div>

          <div className="text-xs text-zinc-500 space-y-1">
            <p className="font-medium text-zinc-400">Quick start:</p>
            <code className="block bg-zinc-900 p-2 rounded text-[10px] font-mono">
              chmod +x masterselects-helper<br />
              ./masterselects-helper
            </code>
          </div>

          <button
            onClick={onRetry}
            disabled={checking}
            className="w-full text-center text-xs text-zinc-400 hover:text-white py-1 transition-colors disabled:opacity-50"
          >
            {checking ? 'Checking...' : 'Retry Connection'}
          </button>
        </div>
      )}

      {/* Disabled state */}
      {!turboModeEnabled && (
        <p className="text-xs text-zinc-500">
          Enable Turbo Mode to use hardware-accelerated decoding for professional codecs.
        </p>
      )}
    </div>
  );
}

export default NativeHelperStatus;
