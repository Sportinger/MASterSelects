// AI Tools Service - Modular architecture
// Provides tools for AI chat to control timeline editing
// Uses OpenAI function calling format

import { Logger } from '../logger';
import { useTimelineStore } from '../../stores/timeline';

const log = Logger.create('AITool');
import { useMediaStore } from '../../stores/mediaStore';
import { startBatch, endBatch } from '../../stores/historyStore';
import type { ToolResult } from './types';
import { MODIFYING_TOOLS } from './types';
import { executeToolInternal } from './handlers';

// Re-export types
export type { ToolResult, ToolDefinition } from './types';
export { MODIFYING_TOOLS } from './types';

// Re-export tool definitions
export { AI_TOOLS } from './definitions';
export {
  timelineToolDefinitions,
  clipToolDefinitions,
  trackToolDefinitions,
  analysisToolDefinitions,
  previewToolDefinitions,
  mediaToolDefinitions,
} from './definitions';

// Re-export utilities
export { getQuickTimelineSummary, formatClipInfo, formatTrackInfo, captureFrameGrid } from './utils';

// Re-export handlers for advanced usage
export { executeToolInternal } from './handlers';

/**
 * Execute an AI tool with history tracking
 * Main entry point for AI chat integration
 */
export async function executeAITool(toolName: string, args: Record<string, unknown>): Promise<ToolResult> {
  const timelineStore = useTimelineStore.getState();
  const mediaStore = useMediaStore.getState();

  // Track history for modifying operations
  const isModifying = MODIFYING_TOOLS.has(toolName);
  if (isModifying) {
    startBatch(`AI: ${toolName}`);
  }

  try {
    const result = await executeToolInternal(toolName, args, timelineStore, mediaStore);

    if (isModifying) {
      endBatch();
    }

    return result;
  } catch (error) {
    if (isModifying) {
      endBatch();
    }
    log.error(`Error executing ${toolName}`, error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}
