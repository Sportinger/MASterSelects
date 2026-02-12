// AI Tools Types

export interface ToolResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

// Tools that modify the timeline or media (need history tracking)
export const MODIFYING_TOOLS = new Set([
  'splitClip', 'splitClipEvenly', 'splitClipAtTimes', 'deleteClip', 'deleteClips', 'moveClip', 'trimClip',
  'createTrack', 'deleteTrack', 'setTrackVisibility', 'setTrackMuted',
  'cutRangesFromClip',
  // Media tools
  'createMediaFolder', 'renameMediaItem', 'deleteMediaItem', 'moveMediaItems',
  'createComposition',
  'executeBatch',
]);

// Tool definition type (OpenAI function calling format)
export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, unknown>;
      required: string[];
    };
  };
}
