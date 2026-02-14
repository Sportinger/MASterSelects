// Feature flags for engine subsystems still in development.
// Scene Graph, Dirty Tracking, Structural Sharing are always-on (no flag needed).
// Toggle at runtime via: window.__ENGINE_FLAGS__

export const flags = {
  useRenderGraph: false,  // Render Graph executor (stubs — not ready)
  useDecoderPool: false,  // Shared decoder pool (not wired yet)
};

// Expose for runtime toggling from devtools
if (typeof window !== 'undefined') {
  (window as any).__ENGINE_FLAGS__ = flags;
}
