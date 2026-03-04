// AI execution state - separated to avoid circular imports
// between aiTools/index.ts and handlers/clips.ts

let _aiExecutionActive = false;

export function setAIExecutionActive(active: boolean): void {
  _aiExecutionActive = active;
}

export function isAIExecutionActive(): boolean {
  return _aiExecutionActive;
}
