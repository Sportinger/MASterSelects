/**
 * Professional Logger Service
 *
 * Features:
 * - Log levels: DEBUG, INFO, WARN, ERROR
 * - Module-based filtering via localStorage
 * - Timestamps and stack traces
 * - In-memory log buffer for AI agent inspection
 * - Console helpers for runtime control
 *
 * Usage:
 *   import { Logger } from '@/services/logger';
 *   const log = Logger.create('WebGPUEngine');
 *   log.debug('Device created');
 *   log.info('Render started');
 *   log.warn('Fallback activated');
 *   log.error('Device lost', error);
 *
 * Console Commands:
 *   Logger.enable('*')                    // Enable all modules
 *   Logger.enable('WebGPU,FFmpeg')        // Enable specific modules
 *   Logger.disable()                      // Disable all debug logs
 *   Logger.setLevel('DEBUG')              // Set minimum log level
 *   Logger.getBuffer()                    // Get recent logs (for AI inspection)
 *   Logger.getBuffer('error')             // Get only errors
 *   Logger.search('device')               // Search logs by keyword
 *   Logger.modules()                      // List all registered modules
 *   Logger.status()                       // Show current config
 */

// ============================================================================
// Types
// ============================================================================

export type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  module: string;
  message: string;
  data?: unknown;
  stack?: string;
}

interface LoggerConfig {
  enabled: string[];      // Modules to show debug logs for ('*' = all)
  level: LogLevel;        // Minimum level to display
  timestamps: boolean;    // Show timestamps
  bufferSize: number;     // Max entries in memory buffer
}

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = 'logger_config';
const LOG_LEVELS: Record<LogLevel, number> = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
};

const LEVEL_COLORS: Record<LogLevel, string> = {
  DEBUG: '#888',
  INFO: '#4a9eff',
  WARN: '#ffaa00',
  ERROR: '#ff4444',
};

const LEVEL_ICONS: Record<LogLevel, string> = {
  DEBUG: '🔍',
  INFO: 'ℹ️',
  WARN: '⚠️',
  ERROR: '❌',
};

// ============================================================================
// State
// ============================================================================

const logBuffer: LogEntry[] = [];
const registeredModules = new Set<string>();

let config: LoggerConfig = {
  enabled: [],
  level: 'INFO',
  timestamps: true,
  bufferSize: 500,
};

// Load config from localStorage
function loadConfig(): void {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      config = { ...config, ...parsed };
    }
  } catch {
    // Ignore parse errors
  }
}

function saveConfig(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // Ignore storage errors
  }
}

// Initialize on load
loadConfig();

// ============================================================================
// Core Logger Class
// ============================================================================

class ModuleLogger {
  constructor(private module: string) {
    registeredModules.add(module);
  }

  debug(message: string, data?: unknown): void {
    this.log('DEBUG', message, data);
  }

  info(message: string, data?: unknown): void {
    this.log('INFO', message, data);
  }

  warn(message: string, data?: unknown): void {
    this.log('WARN', message, data);
  }

  error(message: string, error?: unknown): void {
    this.log('ERROR', message, error);
  }

  /** Log with timing - returns a function to call when done */
  time(label: string): () => void {
    const start = performance.now();
    this.debug(`${label} started`);
    return () => {
      const duration = (performance.now() - start).toFixed(2);
      this.debug(`${label} completed in ${duration}ms`);
    };
  }

  /** Log a group of related messages */
  group(label: string, fn: () => void): void {
    if (this.shouldLog('DEBUG')) {
      console.group(`[${this.module}] ${label}`);
      fn();
      console.groupEnd();
    } else {
      fn();
    }
  }

  private log(level: LogLevel, message: string, data?: unknown): void {
    const entry = this.createEntry(level, message, data);
    this.addToBuffer(entry);

    if (!this.shouldLog(level)) return;

    const prefix = this.formatPrefix(entry);
    const args: unknown[] = [prefix, message];

    if (data !== undefined) {
      args.push(data);
    }

    switch (level) {
      case 'DEBUG':
        console.debug(...args);
        break;
      case 'INFO':
        console.info(...args);
        break;
      case 'WARN':
        console.warn(...args);
        break;
      case 'ERROR':
        console.error(...args);
        if (data instanceof Error && data.stack) {
          console.error(data.stack);
        }
        break;
    }
  }

  private shouldLog(level: LogLevel): boolean {
    // Always log errors
    if (level === 'ERROR') return true;

    // Check level threshold
    if (LOG_LEVELS[level] < LOG_LEVELS[config.level]) return false;

    // For DEBUG, check if module is enabled
    if (level === 'DEBUG') {
      if (config.enabled.length === 0) return false;
      if (config.enabled.includes('*')) return true;
      return config.enabled.some(pattern =>
        this.module.toLowerCase().includes(pattern.toLowerCase())
      );
    }

    return true;
  }

  private createEntry(level: LogLevel, message: string, data?: unknown): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      module: this.module,
      message,
    };

    if (data !== undefined) {
      if (data instanceof Error) {
        entry.data = { name: data.name, message: data.message };
        entry.stack = data.stack;
      } else {
        entry.data = data;
      }
    }

    return entry;
  }

  private formatPrefix(entry: LogEntry): string {
    const parts: string[] = [];

    if (config.timestamps) {
      const time = entry.timestamp.split('T')[1].split('.')[0];
      parts.push(`%c${time}`);
    }

    parts.push(`%c[${this.module}]`);
    parts.push(`%c${LEVEL_ICONS[entry.level]}`);

    // Return with color styling for console
    const timeStyle = 'color: #666';
    const moduleStyle = `color: ${LEVEL_COLORS[entry.level]}; font-weight: bold`;
    const iconStyle = '';

    return parts.join(' ').replace(/%c/g, '') + ' ';
  }

  private addToBuffer(entry: LogEntry): void {
    logBuffer.push(entry);
    if (logBuffer.length > config.bufferSize) {
      logBuffer.shift();
    }
  }
}

// ============================================================================
// Static Logger API
// ============================================================================

export const Logger = {
  /**
   * Create a logger for a specific module
   */
  create(module: string): ModuleLogger {
    return new ModuleLogger(module);
  },

  /**
   * Enable debug logging for modules
   * @param modules - Comma-separated module names or '*' for all
   */
  enable(modules: string = '*'): void {
    config.enabled = modules === '*' ? ['*'] : modules.split(',').map(m => m.trim());
    saveConfig();
    console.log(`[Logger] Debug enabled for: ${modules}`);
  },

  /**
   * Disable all debug logging
   */
  disable(): void {
    config.enabled = [];
    saveConfig();
    console.log('[Logger] Debug logging disabled');
  },

  /**
   * Set minimum log level
   */
  setLevel(level: LogLevel): void {
    config.level = level;
    saveConfig();
    console.log(`[Logger] Level set to: ${level}`);
  },

  /**
   * Toggle timestamps
   */
  setTimestamps(enabled: boolean): void {
    config.timestamps = enabled;
    saveConfig();
  },

  /**
   * Get buffered logs (for AI inspection)
   */
  getBuffer(levelFilter?: LogLevel): LogEntry[] {
    if (!levelFilter) return [...logBuffer];
    const minLevel = LOG_LEVELS[levelFilter];
    return logBuffer.filter(e => LOG_LEVELS[e.level] >= minLevel);
  },

  /**
   * Search logs by keyword
   */
  search(keyword: string): LogEntry[] {
    const lower = keyword.toLowerCase();
    return logBuffer.filter(e =>
      e.message.toLowerCase().includes(lower) ||
      e.module.toLowerCase().includes(lower) ||
      JSON.stringify(e.data || '').toLowerCase().includes(lower)
    );
  },

  /**
   * Get recent errors
   */
  errors(): LogEntry[] {
    return logBuffer.filter(e => e.level === 'ERROR');
  },

  /**
   * List all registered modules
   */
  modules(): string[] {
    return [...registeredModules].sort();
  },

  /**
   * Show current configuration
   */
  status(): void {
    console.log('[Logger] Current Configuration:');
    console.table({
      'Debug Enabled': config.enabled.length > 0 ? config.enabled.join(', ') : 'None',
      'Min Level': config.level,
      'Timestamps': config.timestamps,
      'Buffer Size': config.bufferSize,
      'Buffer Used': logBuffer.length,
      'Registered Modules': registeredModules.size,
    });
  },

  /**
   * Clear the log buffer
   */
  clear(): void {
    logBuffer.length = 0;
    console.log('[Logger] Buffer cleared');
  },

  /**
   * Export logs as JSON (for AI analysis)
   */
  export(): string {
    return JSON.stringify({
      config,
      modules: [...registeredModules],
      logs: logBuffer,
    }, null, 2);
  },

  /**
   * Pretty print recent logs to console
   */
  dump(count: number = 50): void {
    const recent = logBuffer.slice(-count);
    console.log(`[Logger] Last ${recent.length} entries:`);
    recent.forEach(e => {
      const time = e.timestamp.split('T')[1].split('.')[0];
      const color = LEVEL_COLORS[e.level];
      console.log(
        `%c${time} %c[${e.module}] %c${e.level}: ${e.message}`,
        'color: #666',
        `color: ${color}; font-weight: bold`,
        `color: ${color}`,
        e.data || ''
      );
    });
  },

  /**
   * Get a summary for AI agents
   */
  summary(): {
    totalLogs: number;
    errorCount: number;
    warnCount: number;
    recentErrors: LogEntry[];
    activeModules: string[];
  } {
    const errors = logBuffer.filter(e => e.level === 'ERROR');
    const warns = logBuffer.filter(e => e.level === 'WARN');
    const recentModules = new Set(logBuffer.slice(-100).map(e => e.module));

    return {
      totalLogs: logBuffer.length,
      errorCount: errors.length,
      warnCount: warns.length,
      recentErrors: errors.slice(-10),
      activeModules: [...recentModules],
    };
  },
};

// ============================================================================
// Global Access (for console debugging)
// ============================================================================

if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).Logger = Logger;

  // ============================================================================
  // Browser Log Bridge - Auto-sync logs to dev server for AI agent access
  // ============================================================================
  let syncInterval: number | null = null;

  const startLogSync = () => {
    if (syncInterval) return;

    const sync = () => {
      try {
        const summary = Logger.summary();
        // Add recent logs for more context
        const recentLogs = logBuffer.slice(-100);
        const payload = { ...summary, recentLogs };

        fetch('/api/logs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }).catch(() => {}); // Silently fail if dev server not available
      } catch {
        // Ignore errors
      }
    };

    // Sync immediately and then every 2 seconds
    sync();
    syncInterval = window.setInterval(sync, 2000);
  };

  const stopLogSync = () => {
    if (syncInterval) {
      clearInterval(syncInterval);
      syncInterval = null;
    }
  };

  // Auto-start in development
  if (import.meta.env.DEV) {
    startLogSync();
  }

  // Expose controls
  (window as unknown as Record<string, unknown>).LogSync = {
    start: startLogSync,
    stop: stopLogSync,
    status: () => syncInterval ? 'running' : 'stopped',
  };
}

// ============================================================================
// Convenience Exports
// ============================================================================

/** Quick access to create a logger */
export const createLogger = Logger.create;

/** Default logger for quick one-off logs */
export const log = Logger.create('App');
