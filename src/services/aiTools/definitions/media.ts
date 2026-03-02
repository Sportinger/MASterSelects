// Media Panel Tool Definitions

import type { ToolDefinition } from '../types';

export const mediaToolDefinitions: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'getMediaItems',
      description: 'Get all items in the media panel: files (video, audio, image), compositions, and folders. Useful for understanding project structure.',
      parameters: {
        type: 'object',
        properties: {
          folderId: {
            type: 'string',
            description: 'Get items in a specific folder. Omit or null for root level items.',
          },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'createMediaFolder',
      description: 'Create a new folder in the media panel for organizing files.',
      parameters: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Name of the new folder',
          },
          parentFolderId: {
            type: 'string',
            description: 'ID of parent folder (omit for root level)',
          },
        },
        required: ['name'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'renameMediaItem',
      description: 'Rename a media item (file, folder, or composition).',
      parameters: {
        type: 'object',
        properties: {
          itemId: {
            type: 'string',
            description: 'ID of the item to rename',
          },
          newName: {
            type: 'string',
            description: 'New name for the item',
          },
        },
        required: ['itemId', 'newName'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'deleteMediaItem',
      description: 'Delete a media item (file, folder, or composition). Warning: Folders delete all contents.',
      parameters: {
        type: 'object',
        properties: {
          itemId: {
            type: 'string',
            description: 'ID of the item to delete',
          },
        },
        required: ['itemId'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'moveMediaItems',
      description: 'Move media items to a different folder.',
      parameters: {
        type: 'object',
        properties: {
          itemIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of item IDs to move',
          },
          targetFolderId: {
            type: 'string',
            description: 'ID of target folder (omit or null to move to root)',
          },
        },
        required: ['itemIds'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'createComposition',
      description: 'Create a new composition (timeline sequence).',
      parameters: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Name of the composition',
          },
          width: {
            type: 'number',
            description: 'Width in pixels (default: 1920)',
          },
          height: {
            type: 'number',
            description: 'Height in pixels (default: 1080)',
          },
          frameRate: {
            type: 'number',
            description: 'Frame rate (default: 30)',
          },
          duration: {
            type: 'number',
            description: 'Duration in seconds (default: 60)',
          },
        },
        required: ['name'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'selectMediaItems',
      description: 'Select items in the media panel.',
      parameters: {
        type: 'object',
        properties: {
          itemIds: {
            type: 'array',
            items: { type: 'string' },
            description: 'Array of item IDs to select',
          },
        },
        required: ['itemIds'],
      },
    },
  },
];
