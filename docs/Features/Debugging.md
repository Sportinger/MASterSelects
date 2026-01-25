# Debugging & Logging

MASterSelects includes a professional Logger service designed for both human debugging and AI-assisted development.

## Overview

The Logger service (`src/services/logger.ts`) provides:

| Feature | Description |
|---------|-------------|
| **Log Levels** | DEBUG, INFO, WARN, ERROR with level filtering |
| **Module Filtering** | Enable debug logs for specific modules only |
| **In-Memory Buffer** | 500 entries stored for inspection |
| **Global Access** | `window.Logger` available in browser console |
| **AI-Agent Support** | Structured data for AI code assistants |
| **Timestamps** | Optional timestamp prefixes |
| **Stack Traces** | Automatic capture for errors |

---

## Console Commands

All commands are available via `window.Logger` or just `Logger` in the browser console.

### Enable/Disable Debug Logs

```javascript
// Enable debug logs for specific modules (comma-separated)
Logger.enable('WebGPU,FFmpeg,Export')

// Enable all debug logs
Logger.enable('*')

// Disable debug logs (errors still shown)
Logger.disable()
```

### Set Log Level

```javascript
// Show all logs (DEBUG and above)
Logger.setLevel('DEBUG')

// Show INFO and above (hide DEBUG)
Logger.setLevel('INFO')

// Show only warnings and errors
Logger.setLevel('WARN')

// Show only errors
Logger.setLevel('ERROR')
```

### Inspect Logs

```javascript
// Get all buffered logs
Logger.getBuffer()

// Get only errors
Logger.getBuffer('ERROR')

// Get only warnings and errors
Logger.getBuffer('WARN')

// Search logs by keyword
Logger.search('device')
Logger.search('export')

// Get recent errors only
Logger.errors()

// Pretty print last N entries
Logger.dump(50)

// Get summary for AI agents
Logger.summary()
// Returns: { totalLogs, errorCount, warnCount, recentErrors, activeModules }

// Export all logs as JSON
Logger.export()
```

### Status & Configuration

```javascript
// Show current configuration
Logger.status()
// Output:
// [Logger] Current Configuration:
// ┌─────────────────────┬───────────────────────┐
// │ Debug Enabled       │ WebGPU, FFmpeg        │
// │ Min Level           │ INFO                  │
// │ Timestamps          │ true                  │
// │ Buffer Size         │ 500                   │
// │ Buffer Used         │ 127                   │
// │ Registered Modules  │ 45                    │
// └─────────────────────┴───────────────────────┘

// List all registered modules
Logger.modules()
// Returns: ['AudioEncoder', 'AudioMixer', 'Compositor', 'Export', ...]

// Clear the log buffer
Logger.clear()

// Toggle timestamps
Logger.setTimestamps(false)
```

---

## Usage in Code

### Basic Usage

```typescript
import { Logger } from '@/services/logger';

// Create a logger for your module
const log = Logger.create('MyModule');

// Log at different levels
log.debug('Verbose debugging info', { data });  // Only shows if DEBUG enabled
log.info('Important event');                     // Always shows (unless level > INFO)
log.warn('Warning message', data);               // Orange in console
log.error('Error occurred', error);              // Red, always shows, captures stack
```

### Timing Helper

```typescript
const log = Logger.create('Export');

// Start timing
const done = log.time('Encoding video');

// ... do work ...

// Log completion with duration
done();
// Output: [Export] Encoding video completed in 1234.56ms
```

### Grouped Logs

```typescript
const log = Logger.create('Compositor');

log.group('Rendering frame 42', () => {
  log.debug('Collecting layers');
  log.debug('Applying effects');
  log.debug('Compositing');
});
// Output is grouped in console when DEBUG enabled
```

---

## Module Naming Convention

Modules are named after their file or class:

| File | Module Name |
|------|-------------|
| `WebGPUEngine.ts` | `WebGPUEngine` |
| `FFmpegBridge.ts` | `FFmpegBridge` |
| `AudioEncoder.ts` | `AudioEncoder` |
| `ProjectCoreService.ts` | `ProjectCore` |
| `Timeline.tsx` | `Timeline` |

### Common Module Groups

```javascript
// GPU/Rendering
Logger.enable('WebGPU,Compositor,RenderLoop,TextureManager')

// Export pipeline
Logger.enable('Export,FrameExporter,VideoEncoder,AudioEncoder,FFmpeg')

// Audio system
Logger.enable('Audio,AudioMixer,AudioEncoder,TimeStretch')

// Project/Storage
Logger.enable('Project,ProjectCore,FileStorage')

// Timeline
Logger.enable('Timeline,Clip,Track,Keyframe')
```

---

## AI-Agent Inspection

The Logger is designed to help AI code assistants (like Claude) understand what's happening in the application.

### Summary for AI

```javascript
const summary = Logger.summary();
// {
//   totalLogs: 234,
//   errorCount: 2,
//   warnCount: 5,
//   recentErrors: [...last 10 errors...],
//   activeModules: ['WebGPUEngine', 'Export', 'FFmpegBridge']
// }
```

### Search for Issues

```javascript
// Find all logs related to a specific issue
Logger.search('device lost')
Logger.search('encode failed')
Logger.search('permission denied')
```

### Export for Analysis

```javascript
// Get full log data as JSON
const logData = Logger.export();
// Contains: config, modules, and all buffered logs
```

---

## Log Entry Structure

Each log entry contains:

```typescript
interface LogEntry {
  timestamp: string;    // ISO timestamp
  level: LogLevel;      // 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  module: string;       // Module name
  message: string;      // Log message
  data?: unknown;       // Optional attached data
  stack?: string;       // Stack trace (for errors)
}
```

---

## Persistence

Logger configuration is saved to `localStorage`:

- `logger_config` - Stores enabled modules, level, timestamps setting

Configuration persists across page refreshes.

---

## Performance Monitoring

In addition to the Logger, MASterSelects includes performance monitoring:

### PerformanceStats (`src/engine/stats/PerformanceStats.ts`)

Tracks:
- Frame rate (FPS)
- RAF gap (requestAnimationFrame latency)
- Texture import time
- Render pass time
- Submit time
- Frame drops and drop reasons

### PerformanceMonitor (`src/services/performanceMonitor.ts`)

- Auto-starts with the application
- Detects slow frames (>100ms threshold)
- Auto-resets quality parameters after 5+ slow frames
- Provides callback system for performance events

---

## Troubleshooting

### Common Debug Scenarios

**Black preview / No rendering:**
```javascript
Logger.enable('WebGPU,Compositor,RenderLoop')
// Check for device issues, texture errors
```

**Export fails:**
```javascript
Logger.enable('Export,FrameExporter,VideoEncoder,FFmpeg')
// Check for encoding errors, codec issues
```

**Audio out of sync:**
```javascript
Logger.enable('Audio,AudioMixer,TimeStretch')
// Check for timing issues
```

**File import problems:**
```javascript
Logger.enable('Media,Import,Project')
// Check for file access, format issues
```

---

## Best Practices

1. **Use appropriate log levels:**
   - `debug` for verbose/frequent logs (disabled by default)
   - `info` for important events
   - `warn` for recoverable issues
   - `error` for failures

2. **Include context data:**
   ```typescript
   log.debug('Frame rendered', { frameNumber, duration, layerCount });
   ```

3. **Use timing for performance:**
   ```typescript
   const done = log.time('Heavy operation');
   // ... work ...
   done();
   ```

4. **Keep module names consistent** with file/class names

5. **Don't log sensitive data** (API keys, user data)

---

*Documentation updated January 2026*
