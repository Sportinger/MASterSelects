// YouTube AI Tool Definitions

import type { ToolDefinition } from '../types';

export const youtubeToolDefinitions: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'searchYouTube',
      description: 'Search YouTube for videos by keyword. Returns video results with title, channel, duration, views, and URL. Results also appear in the Downloads panel. Requires a YouTube API key to be configured in settings.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query string',
          },
          maxResults: {
            type: 'number',
            description: 'Maximum number of results to return (1-20, default 10)',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'listVideoFormats',
      description: 'List available download formats and qualities for a video URL. Works with YouTube, TikTok, Instagram, Twitter/X, Vimeo, and other platforms supported by yt-dlp. Returns recommended formats and detailed format information. Requires the Native Helper to be running.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'Video URL (YouTube, TikTok, Instagram, etc.) or YouTube video ID',
          },
        },
        required: ['url'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'downloadAndImportVideo',
      description: 'Download a video and import it directly into the timeline. Creates a pending clip that shows download progress, then converts to a real playable clip when done. The download can take several minutes depending on video length and quality. Requires the Native Helper to be running.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'Video URL (YouTube, TikTok, Instagram, etc.)',
          },
          title: {
            type: 'string',
            description: 'Title for the clip on the timeline',
          },
          formatId: {
            type: 'string',
            description: 'Format ID from listVideoFormats (optional, uses best quality if not specified)',
          },
          thumbnail: {
            type: 'string',
            description: 'Thumbnail URL for the pending clip (optional)',
          },
        },
        required: ['url', 'title'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'getYouTubeVideos',
      description: 'Get the list of videos currently in the Downloads panel (from previous searches or pasted URLs).',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
];
