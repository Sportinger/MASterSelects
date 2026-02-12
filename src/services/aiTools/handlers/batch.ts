// Batch Execution Handler

import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import type { ToolResult } from '../types';
import { executeToolInternal } from './index';

interface BatchAction {
  tool: string;
  args: Record<string, unknown>;
}

interface BatchActionResult {
  tool: string;
  success: boolean;
  data?: unknown;
  error?: string;
}

/**
 * Execute multiple tools in sequence as a single batch.
 * Re-fetches fresh store state between actions so that
 * clip IDs from splits are available to subsequent actions.
 */
export async function handleExecuteBatch(
  args: Record<string, unknown>
): Promise<ToolResult> {
  const actions = args.actions as BatchAction[];

  if (!actions || !Array.isArray(actions) || actions.length === 0) {
    return { success: false, error: 'actions must be a non-empty array' };
  }

  const results: BatchActionResult[] = [];
  let allSucceeded = true;

  for (const action of actions) {
    // Re-fetch fresh state before each action
    const timelineStore = useTimelineStore.getState();
    const mediaStore = useMediaStore.getState();

    try {
      const result = await executeToolInternal(
        action.tool,
        action.args || {},
        timelineStore,
        mediaStore
      );

      results.push({
        tool: action.tool,
        success: result.success,
        data: result.data,
        error: result.error,
      });

      if (!result.success) {
        allSucceeded = false;
      }
    } catch (error) {
      allSucceeded = false;
      results.push({
        tool: action.tool,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  return {
    success: allSucceeded,
    data: {
      totalActions: actions.length,
      succeeded: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length,
      results,
    },
  };
}
