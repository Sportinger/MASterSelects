// Clip Editing Tool Definitions

import type { ToolDefinition } from '../types';

export const clipToolDefinitions: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'getClipDetails',
      description: 'Get detailed information about a specific clip including its analysis data, transcript, effects, and transform properties.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to get details for',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getClipsInTimeRange',
      description: 'Get all clips that overlap with a specific time range.',
      parameters: {
        type: 'object',
        properties: {
          startTime: {
            type: 'number',
            description: 'Start time in seconds',
          },
          endTime: {
            type: 'number',
            description: 'End time in seconds',
          },
          trackType: {
            type: 'string',
            enum: ['video', 'audio', 'all'],
            description: 'Filter by track type (default: all)',
          },
        },
        required: ['startTime', 'endTime'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'splitClip',
      description: 'Split a clip at a specific time, creating two separate clips.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to split',
          },
          splitTime: {
            type: 'number',
            description: 'The time in seconds (timeline time, not clip-relative) where to split',
          },
        },
        required: ['clipId', 'splitTime'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'deleteClip',
      description: 'Delete a clip from the timeline.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to delete',
          },
        },
        required: ['clipId'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'deleteClips',
      description: 'Delete multiple clips from the timeline at once.',
      parameters: {
        type: 'object',
        properties: {
          clipIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of clip IDs to delete',
          },
        },
        required: ['clipIds'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'moveClip',
      description: 'Move a clip to a new position and/or track.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to move',
          },
          newStartTime: {
            type: 'number',
            description: 'New start time in seconds',
          },
          newTrackId: {
            type: 'string',
            description: 'ID of the track to move the clip to (optional, keeps current track if not specified)',
          },
        },
        required: ['clipId', 'newStartTime'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'trimClip',
      description: 'Trim a clip by adjusting its in and out points (relative to the source media).',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to trim',
          },
          inPoint: {
            type: 'number',
            description: 'New in point in seconds (relative to source media start)',
          },
          outPoint: {
            type: 'number',
            description: 'New out point in seconds (relative to source media start)',
          },
        },
        required: ['clipId', 'inPoint', 'outPoint'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'cutRangesFromClip',
      description: 'Cut out multiple time ranges from a clip. This is the preferred way to remove multiple sections (like all low-focus parts). It handles clip ID changes automatically by processing from end to start.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to edit',
          },
          ranges: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                timelineStart: { type: 'number', description: 'Start time on timeline (seconds)' },
                timelineEnd: { type: 'number', description: 'End time on timeline (seconds)' },
              },
              required: ['timelineStart', 'timelineEnd'],
            },
            description: 'Array of time ranges to cut out (in timeline time). Use the timelineStart/timelineEnd values from findLowQualitySections.',
          },
        },
        required: ['clipId', 'ranges'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'splitClipEvenly',
      description: 'Split a clip into N equal parts. This is much faster than using executeBatch with individual splitClip calls. Handles linked audio automatically.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to split',
          },
          parts: {
            type: 'number',
            description: 'Number of equal parts to split into (minimum 2)',
          },
        },
        required: ['clipId', 'parts'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'splitClipAtTimes',
      description: 'Split a clip at multiple specific times in a single operation. This is much faster than using executeBatch with individual splitClip calls. Handles linked audio and changing clip IDs automatically.',
      parameters: {
        type: 'object',
        properties: {
          clipId: {
            type: 'string',
            description: 'The ID of the clip to split',
          },
          times: {
            type: 'array',
            items: { type: 'number' },
            description: 'Array of timeline times (in seconds) where to split the clip',
          },
        },
        required: ['clipId', 'times'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'selectClips',
      description: 'Select one or more clips in the timeline.',
      parameters: {
        type: 'object',
        properties: {
          clipIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of clip IDs to select',
          },
        },
        required: ['clipIds'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'clearSelection',
      description: 'Clear the current clip selection.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
];
