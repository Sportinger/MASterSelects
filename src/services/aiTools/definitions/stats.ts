import type { ToolDefinition } from '../types';

export const statsToolDefinitions: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'getStats',
      description: 'Get current engine/playback stats snapshot for debugging. Returns FPS, timing breakdown, decoder info, drops, playback health, audio status, GPU info.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getStatsHistory',
      description: 'Collect multiple stats snapshots over a time window for performance analysis. Returns an array of timestamped samples.',
      parameters: {
        type: 'object',
        properties: {
          samples: { type: 'number', description: 'Number of samples to collect (default: 5, max: 30)' },
          intervalMs: { type: 'number', description: 'Milliseconds between samples (default: 200, min: 100)' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getLogs',
      description: 'Get recent buffered browser logs for debugging. Supports filtering by level, module name, and search text.',
      parameters: {
        type: 'object',
        properties: {
          limit: { type: 'number', description: 'Maximum number of recent log entries to return (default: 100, max: 500)' },
          level: { type: 'string', description: 'Minimum log level filter: DEBUG, INFO, WARN, ERROR' },
          module: { type: 'string', description: 'Substring filter for the logger module name, e.g. PlaybackHealth or CutTransition' },
          search: { type: 'string', description: 'Substring filter against the message and serialized data' },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getPlaybackTrace',
      description: 'Get raw recent playback pipeline events and health state for debugging WebCodecs/VF playback issues.',
      parameters: {
        type: 'object',
        properties: {
          windowMs: { type: 'number', description: 'Time window in milliseconds to inspect (default: 5000, max: 30000)' },
          limit: { type: 'number', description: 'Maximum number of recent WC/VF events to include (default: 120, max: 500)' },
        },
        required: [],
      },
    },
  },
];
