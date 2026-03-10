export type ScrubSettleStage = 'settle' | 'retry' | 'warmup';
export type ScrubSettleReason = 'manual-seek' | 'scrub-stop' | 'playback-stop';

export interface ScrubSettleEntry {
  clipId: string;
  targetTime: number;
  stage: ScrubSettleStage;
  reason: ScrubSettleReason;
  deadlineAt: number;
}

class ScrubSettleState {
  private entries = new Map<string, ScrubSettleEntry>();

  begin(clipId: string, targetTime: number, timeoutMs: number, reason?: ScrubSettleReason): void {
    const previous = this.entries.get(clipId);
    this.entries.set(clipId, {
      clipId,
      targetTime,
      stage: 'settle',
      reason: reason ?? previous?.reason ?? 'manual-seek',
      deadlineAt: performance.now() + timeoutMs,
    });
  }

  markRetry(clipId: string, targetTime: number, timeoutMs: number, reason?: ScrubSettleReason): void {
    const previous = this.entries.get(clipId);
    this.entries.set(clipId, {
      clipId,
      targetTime,
      stage: 'retry',
      reason: reason ?? previous?.reason ?? 'manual-seek',
      deadlineAt: performance.now() + timeoutMs,
    });
  }

  markWarmup(clipId: string, targetTime: number, timeoutMs: number, reason?: ScrubSettleReason): void {
    const previous = this.entries.get(clipId);
    this.entries.set(clipId, {
      clipId,
      targetTime,
      stage: 'warmup',
      reason: reason ?? previous?.reason ?? 'manual-seek',
      deadlineAt: performance.now() + timeoutMs,
    });
  }

  get(clipId?: string): ScrubSettleEntry | undefined {
    if (!clipId) {
      return undefined;
    }
    return this.entries.get(clipId);
  }

  isPending(clipId?: string): boolean {
    return !!clipId && this.entries.has(clipId);
  }

  isDue(clipId?: string): boolean {
    const entry = this.get(clipId);
    return !!entry && performance.now() >= entry.deadlineAt;
  }

  resolve(clipId?: string): void {
    if (!clipId) {
      return;
    }
    this.entries.delete(clipId);
  }

  clear(): void {
    this.entries.clear();
  }
}

export const scrubSettleState = new ScrubSettleState();
