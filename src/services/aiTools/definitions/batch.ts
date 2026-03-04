// Batch Execution Tool Definition

import type { ToolDefinition } from '../types';

export const batchToolDefinitions: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'executeBatch',
      description: 'Execute multiple timeline/media actions in sequence as a single batch. Use this for efficiency when you need to perform multiple operations (e.g. multiple splits, delete + move, etc.). All actions share a single undo point. Important: each action gets fresh state, so clip IDs from earlier splits are available to later actions.',
      parameters: {
        type: 'object',
        properties: {
          actions: {
            type: 'array',
            description: 'Array of actions to execute in order',
            items: {
              type: 'object',
              properties: {
                tool: {
                  type: 'string',
                  description: 'The tool name to execute (e.g. splitClip, deleteClip, moveClip)',
                },
                args: {
                  type: 'object',
                  description: 'Arguments for the tool',
                },
              },
              required: ['tool', 'args'],
            },
          },
          staggerDelayMs: {
            type: 'number',
            description: 'Delay between actions in ms for visual stagger effect (default: 100, set to 0 for instant)',
          },
        },
        required: ['actions'],
      },
    },
  },
];
