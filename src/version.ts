// App version - INCREMENT ON EVERY COMMIT!
// Format: MAJOR.MINOR.PATCH
// Increment PATCH (0.0.X) for each commit
export const APP_VERSION = '1.1.7';

// Build/Platform notice shown at top of changelog (set to null to hide)
export const BUILD_NOTICE: {
  type: 'info' | 'warning' | 'success';
  title: string;
  message: string;
} | null = null;

// Change entry type
export interface ChangeEntry {
  type: 'new' | 'fix' | 'improve';
  title: string;
  description?: string;
  section?: string; // Optional section header to create visual dividers
  commit?: string; // Git commit hash for linking to GitHub
}

// Time-grouped changelog entry
export interface TimeGroupedChanges {
  label: string; // "Today", "Last Week", "This Month", "Earlier"
  dateRange: string; // "Jan 20" or "Jan 13-19" etc
  changes: ChangeEntry[];
}

// Calculate relative time labels based on current date
function getTimeLabel(date: Date): { label: string; sortOrder: number } {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  if (date >= today) {
    return { label: 'Today', sortOrder: 0 };
  } else if (date >= yesterday) {
    return { label: 'Yesterday', sortOrder: 1 };
  } else if (date >= weekAgo) {
    return { label: 'Last Week', sortOrder: 2 };
  } else if (date >= monthAgo) {
    return { label: 'This Month', sortOrder: 3 };
  } else {
    return { label: 'Earlier', sortOrder: 4 };
  }
}

// Raw changelog data with dates
interface RawChangeEntry extends ChangeEntry {
  date: string; // ISO date string YYYY-MM-DD
  section?: string; // Optional section header to create visual dividers
}

const RAW_CHANGELOG: RawChangeEntry[] = [
  // === Feb 3, 2026 - Multi-Select & FCP XML ===
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Multi-Select Clip Movement',
    description: 'Select multiple clips with Shift+Click and drag them together - clips move as a group with proper boundary collision',
    commit: '6f3eaa3',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Multi-Select Keyframe Movement',
    description: 'Select multiple keyframes and move them together by the same time delta',
    commit: '6f3eaa3',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'FCP XML Export',
    description: 'Export timeline to Final Cut Pro XML format for interchange with other NLEs like Premiere and DaVinci Resolve',
    commit: '901843e',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Audio Detection for OBS Recordings',
    description: 'Fixed audio tracks not detected in OBS recordings and other MP4/MOV files - now uses MP4Box for container parsing',
    commit: '52e76b9',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Clip Entrance Animation',
    description: 'Fixed entrance animation showing on wrong actions - now only triggers when switching compositions',
    commit: 'ce1e782',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Snapping Without Alt Key',
    description: 'Fixed snapping not working when enabled - snapping now works by default, Alt temporarily disables it',
    commit: '9dba94e',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Multi-Select Visual Preview',
    description: 'All selected clips now show visual preview during drag, not just on release',
    commit: '7c71257',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Multi-Select Clip Trimming',
    description: 'Fixed clips being accidentally trimmed when moving multiple selected clips together',
    commit: '14137b8',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Multi-Select Group Boundaries',
    description: 'Whole group now stops when any selected clip hits an obstacle - clips cannot be pushed into each other',
    commit: 'e938874',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Audio/Video Sync During Multi-Drag',
    description: 'Fixed video and audio getting out of sync when rapidly dragging multiple selected clips',
    commit: 'f3bb637',
  },

  // === Feb 2, 2026 - Export & Audio Fixes ===
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Export Crash with Empty Frames',
    description: 'Fixed crash when export starts at time 0 but clips begin later - now properly clears export canvas',
    commit: 'c3fc31b',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Device Loss After Export',
    description: 'Fixed WebGPU device crash after export caused by VideoFrames not being closed properly',
    commit: '72dffc3',
  },
  {
    date: '2026-02-02',
    type: 'improve',
    title: 'Export Fallback for Parallel Decode',
    description: 'Export now falls back to HTMLVideoElement when parallel decode fails instead of crashing',
    commit: '6b0c7e0',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Proxy Stuck at 100%',
    description: 'Proxy generation status now updates to ready immediately after frames complete',
    commit: 'ce2b09f',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Nested Composition Visibility',
    description: 'Fixed nested compositions with multiple layers not displaying correctly',
    commit: 'ce2b09f',
  },
  {
    date: '2026-02-02',
    type: 'improve',
    title: 'Scale Keyframe UI',
    description: 'Simplified Scale keyframe controls - single toggle for X and Y',
    commit: '89647b1',
  },
  {
    date: '2026-02-02',
    type: 'improve',
    title: 'EQ Keyframe UI',
    description: 'Simplified EQ keyframe controls - single toggle for all bands',
    commit: '15af792',
  },
  {
    date: '2026-02-02',
    type: 'new',
    title: 'Live EQ via Web Audio',
    description: 'Real-time audio equalization using Web Audio API - hear changes instantly',
    commit: 'b232022',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Audio Track Overlap',
    description: 'Dropping multiple clips now creates new audio tracks when overlap occurs',
    commit: '9fcf9ad',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Audio Volume Range Error',
    description: 'Fixed preview freeze caused by audio volume exceeding valid range',
    commit: '4dc9790',
  },
  {
    date: '2026-02-02',
    type: 'new',
    title: 'Audio Tab for Video Clips',
    description: 'Video clips now have dedicated Audio tab with volume controls and keyframes',
    commit: 'e1c1d15',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Split Clip Audio Elements',
    description: 'Splitting audio-only clips or nested compositions now creates independent audio elements for each part',
    commit: '8c5ace5',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Texture Lifecycle Management',
    description: 'Fix "Destroyed texture used in a submit" warnings causing black preview when switching compositions',
    commit: 'f69d0c7',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Masks After Page Refresh',
    description: 'Masks now show correctly after page refresh and when switching compositions',
    commit: '7331a9f',
  },
  {
    date: '2026-02-02',
    type: 'new',
    title: 'Text Items in Media Panel',
    description: 'Add text clips via Media Panel with drag-to-timeline support',
    commit: 'd822843',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Export Frame Tolerance',
    description: 'Fixed export failing when frame not within tolerance',
    commit: '24445ab',
  },
  {
    date: '2026-02-02',
    type: 'improve',
    title: 'FPS Stats Accuracy',
    description: 'Show render time in stats, fix precision and multiple RAF loop issues',
    commit: '9c61956',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Effects Bleeding Through Layers',
    description: 'Effects now only affect their own layer, not layers below - added effect pre-processing pipeline',
    commit: '889448a',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Nested Comp Transforms When Paused',
    description: 'Nested composition transforms and effects now show correctly when timeline is paused',
    commit: 'b41c8c0',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Nested Comp Keyframes on Load',
    description: 'Nested composition keyframes now load correctly when opening a project',
    commit: '9e4d7fd',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Video Frame Jumping',
    description: 'Reduced video frame jumping when changing effects - use cached frames during seeking',
    commit: '31d205d',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Scrubbing Not Working',
    description: 'Fixed scrubbing and black preview on reload - render loop now stays active',
    commit: '333d3f6',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Audio Decode Spam',
    description: 'Stop spamming audio decode errors for videos without audio track',
    commit: '75f1f91',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Clip Split Playback',
    description: 'Fixed playback issues after cutting clips - now creates independent video/audio elements',
    commit: '51c1c86',
  },
  {
    date: '2026-02-02',
    type: 'improve',
    title: 'Cut Mode Auto-Reset',
    description: 'Cut tool automatically returns to select mode after cutting a clip',
    commit: 'de80778',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Nested Comp Export',
    description: 'Fixed opacity and keyframe animations not applied during export of nested compositions',
  },
  {
    date: '2026-02-02',
    type: 'fix',
    title: 'Keyframe Interpolation',
    description: 'Fixed nested clip keyframes not interpolating correctly - now lookups directly from store',
  },
  {
    date: '2026-02-02',
    type: 'new',
    title: 'Windows Build Notice',
    description: 'Changelog now shows platform compatibility notice at the top',
  },

  // === Feb 1, 2026 - Big Feature Day ===
  {
    date: '2026-02-01',
    type: 'new',
    title: 'Nested Comp Boundary Markers',
    description: 'Visual markers show where clips start/end within nested compositions',
  },
  {
    date: '2026-02-01',
    type: 'improve',
    title: 'ESLint Cleanup',
    description: 'Fixed all ESLint errors for Cloudflare builds, improved type safety',
  },

  // === Jan 31, 2026 - WYSIWYG Thumbnails ===
  {
    date: '2026-01-31',
    type: 'new',
    title: 'WYSIWYG Clip Thumbnails',
    description: 'Thumbnails now show effects applied to clips',
  },
  {
    date: '2026-01-31',
    type: 'improve',
    title: 'Content-Aware Thumbnail Sampling',
    description: 'Thumbnails sample at clip boundaries for better preview coverage',
  },
  {
    date: '2026-01-31',
    type: 'new',
    title: 'Copy/Paste Clips',
    description: 'Ctrl+C/V to copy and paste timeline clips with undo support',
  },
  {
    date: '2026-01-31',
    type: 'fix',
    title: 'Track Header Scroll/Zoom',
    description: 'Disabled accidental Alt+Wheel zoom and Shift+Wheel scroll over track headers',
  },

  // === Jan 30, 2026 - Nested Comp Thumbnails ===
  {
    date: '2026-01-30',
    type: 'new',
    title: 'WebGPU Thumbnail Renderer',
    description: 'GPU-accelerated thumbnail generation for nested compositions',
  },
  {
    date: '2026-01-30',
    type: 'improve',
    title: 'Smart Thumbnail Updates',
    description: 'Only regenerate nested comp thumbnails when content actually changes',
  },

  // === Jan 29, 2026 - Export System V2 ===
  {
    date: '2026-01-29',
    type: 'new',
    title: 'Export System V2',
    description: 'Shared decoder pool with intelligent frame caching for faster exports',
  },
  {
    date: '2026-01-29',
    type: 'new',
    title: 'Export Planner',
    description: 'Smart decode scheduling optimizes export performance',
  },
  {
    date: '2026-01-29',
    type: 'fix',
    title: 'Export Keyframe Seeking',
    description: 'Fixed decode seeking to wrong keyframe position',
  },

  // === Jan 28, 2026 - Nested Comp Fixes ===
  {
    date: '2026-01-28',
    type: 'fix',
    title: 'Nested Comp Layer Order',
    description: 'Fixed layer ordering in nested compositions',
  },
  {
    date: '2026-01-28',
    type: 'fix',
    title: 'Nested Comp Video Playback',
    description: 'Fixed videos not rendering due to readyState timing',
  },
  {
    date: '2026-01-28',
    type: 'improve',
    title: 'Composition Tab UX',
    description: 'Clicking composition tabs now activates Media Panel',
  },

  // === Jan 27, 2026 - Proxy & Caching ===
  {
    date: '2026-01-27',
    type: 'fix',
    title: 'Proxy Scrubbing',
    description: 'Use nearest cached frame as fallback for smoother scrubbing',
  },
  {
    date: '2026-01-27',
    type: 'fix',
    title: 'Backward Seek',
    description: 'Fixed backward seek detection in ParallelDecodeManager',
  },

  // === Jan 26, 2026 - Project Loading & Linux ===
  {
    date: '2026-01-26',
    type: 'fix',
    title: 'Project File Loading',
    description: 'Improved reliability loading files from Raw folder',
  },
  {
    date: '2026-01-26',
    type: 'fix',
    title: 'Linux/Vulkan Preview',
    description: 'Fixed black preview issue on Linux with Vulkan backend',
  },
  {
    date: '2026-01-26',
    type: 'new',
    title: 'IndexedDB Error Dialog',
    description: 'Clear error message when browser storage is corrupted',
  },
  {
    date: '2026-01-26',
    type: 'new',
    title: 'Clip Entrance Animation',
    description: 'Smooth animation when switching compositions',
  },
  {
    date: '2026-01-26',
    type: 'fix',
    title: 'Nested Clip Caching',
    description: 'Optimized nested composition rendering with frame caching',
  },
  {
    date: '2026-01-26',
    type: 'fix',
    title: 'NativeHelper Dialog Crash',
    description: 'Fixed crash with lite helper configuration',
  },

  // === Jan 25, 2026 - MAJOR REFACTORING DAY ===
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'WebGPUEngine Refactor',
    description: 'Split into focused modules - 57% smaller core file',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Timeline Component Refactor',
    description: 'Extract hooks and utilities - 2109 → 1323 lines of code',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'ClipSlice Refactor',
    description: 'Modular clip and helper modules - 66% code reduction',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'FrameExporter Refactor',
    description: 'Split 1510-line file into 8 focused modules',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'ProjectFileService Refactor',
    description: 'Modular architecture with clean separation of concerns',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'AI Tools Refactor',
    description: 'Split monolithic aiTools.ts into modular architecture',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Remove VJ/Mixer Mode',
    description: 'Simplified codebase by removing unused VJ mode entirely',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Codebase Reorganization',
    description: 'Cleaner folder structure and better code organization',
  },
  {
    date: '2026-01-25',
    type: 'new',
    title: 'Marker Drag-to-Create',
    description: 'Drag from M button to create markers with ghost preview',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Markers Extend into Ruler',
    description: 'Markers now visually extend into time ruler like playhead',
  },
  {
    date: '2026-01-25',
    type: 'new',
    title: 'Nested Comp Visual Indicator',
    description: 'Orange outline on nested composition clips for easy identification',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'YouTube Download Organization',
    description: 'Auto-place YouTube downloads in dedicated YouTube folder',
  },
  {
    date: '2026-01-25',
    type: 'fix',
    title: 'Nested Composition Rendering',
    description: 'Fixed WebGPU texture flags causing black preview in nested comps',
  },
  {
    date: '2026-01-25',
    type: 'fix',
    title: 'Nested Comp Audio Handling',
    description: 'Skip audio sync for nested compositions without audio tracks',
  },
  {
    date: '2026-01-25',
    type: 'fix',
    title: 'Export Frame Timing',
    description: 'Handle first frame at non-zero timestamp and clips beyond video duration',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Parallel Decode Error Handling',
    description: 'Strict error reporting instead of silent fallback to HTMLVideoElement',
  },

  // === Jan 24, 2026 ===
  {
    date: '2026-01-24',
    type: 'fix',
    title: 'Blend Mode Instant Update',
    description: 'Blend mode changes now apply immediately in preview',
  },
  {
    date: '2026-01-24',
    type: 'new',
    title: 'Numpad Blend Mode Cycling',
    description: 'Use numpad +/- to cycle through blend modes',
  },
  {
    date: '2026-01-24',
    type: 'fix',
    title: 'Media Thumbnails Persistence',
    description: 'Restore media panel thumbnails after project reload',
  },

  // === Jan 22, 2026 ===
  {
    date: '2026-01-22',
    type: 'fix',
    title: 'Parallel Decode Keyframes',
    description: 'Improved keyframe handling for reliable multi-clip export',
  },
  {
    date: '2026-01-22',
    type: 'fix',
    title: 'Nested Comp Preview',
    description: 'Fix nested composition preview and thumbnail generation',
  },
  {
    date: '2026-01-22',
    type: 'new',
    title: 'Linked Audio Preview',
    description: 'Show linked audio track preview when dragging video',
  },

  // === Jan 21, 2026 ===
  {
    date: '2026-01-21',
    type: 'improve',
    title: 'Removed VJ/Mixer Mode',
    description: 'Simplified codebase by removing unused VJ mode',
  },
  {
    date: '2026-01-21',
    type: 'new',
    title: 'Logger Service',
    description: 'Professional logging system with filtering and search',
  },
  {
    date: '2026-01-21',
    type: 'new',
    title: 'Full Project Persistence',
    description: 'Encrypted API keys, global settings, per-project UI state',
  },
  {
    date: '2026-01-21',
    type: 'new',
    title: 'UI State Persistence',
    description: 'Dock layout and timeline view state saved per composition',
  },
  {
    date: '2026-01-21',
    type: 'new',
    title: 'Automatic Frame Caching',
    description: 'Cache frames during playback for instant scrubbing',
  },
  {
    date: '2026-01-21',
    type: 'fix',
    title: 'B-Frame Decoding',
    description: 'Decode until next keyframe for proper B-frame handling',
  },

  // === Jan 20, 2026 ===
  {
    date: '2026-01-20',
    type: 'new',
    title: 'Export Mode Selection',
    description: 'Choose between WebCodecs Fast, HTMLVideo Precise, or FFmpeg export',
  },
  {
    date: '2026-01-20',
    type: 'new',
    title: 'Audio Master Clock',
    description: 'Playhead follows audio for perfect sync like Premiere/Resolve',
  },
  {
    date: '2026-01-20',
    type: 'new',
    title: 'Varispeed Audio Scrubbing',
    description: 'Continuous playback with speed adjustment while scrubbing',
  },
  {
    date: '2026-01-20',
    type: 'improve',
    title: 'Layer Caching',
    description: 'Better performance when paused or scrubbing',
  },
  {
    date: '2026-01-20',
    type: 'fix',
    title: 'Audio Export Codec Detection',
    description: 'Properly detect AAC/Opus support for audio encoding',
  },
  {
    date: '2026-01-20',
    type: 'improve',
    title: 'WebCodecs Export Optimization',
    description: 'Parallel clip loading and sequential decoding for faster exports',
  },

  // === Jan 13-14, 2026 ===
  {
    date: '2026-01-14',
    type: 'fix',
    title: 'Export Frame Sync',
    description: 'Proper H.264 avcC config and sequential decoding',
  },
  {
    date: '2026-01-13',
    type: 'new',
    title: 'YouTube Video Download',
    description: 'Download YouTube videos via Native Helper with yt-dlp',
  },
  {
    date: '2026-01-13',
    type: 'new',
    title: 'Native Helper App',
    description: 'Hardware-accelerated video codecs for ProRes/DNxHD',
  },
  {
    date: '2026-01-13',
    type: 'new',
    title: 'NativeDecoder Integration',
    description: 'Play professional codecs directly in browser',
  },
  {
    date: '2026-01-13',
    type: 'new',
    title: 'FFmpeg Export with Audio',
    description: 'Full FFmpeg WASM export with progress bar',
  },
  {
    date: '2026-01-13',
    type: 'new',
    title: 'Desktop Mode for Mobile',
    description: 'Option to view full desktop UI on mobile devices',
  },
  {
    date: '2026-01-13',
    type: 'new',
    title: 'Multi-Select Relink Dialog',
    description: 'Batch file relinking for missing media',
  },
  {
    date: '2026-01-13',
    type: 'fix',
    title: 'MOV/MXF Import',
    description: 'Case-insensitive extension detection for all containers',
  },

  // === Jan 12, 2026 ===
  {
    date: '2026-01-12',
    type: 'new',
    title: 'Mobile UI',
    description: 'Touch gestures and responsive layout for mobile devices',
  },
  {
    date: '2026-01-12',
    type: 'new',
    title: 'Relink Dialog',
    description: 'Reconnect missing media files easily',
  },
  {
    date: '2026-01-12',
    type: 'fix',
    title: 'Keyframes Persistence',
    description: 'Keyframes now properly saved and restored in projects',
  },
  {
    date: '2026-01-12',
    type: 'fix',
    title: 'NVIDIA GPU Compatibility',
    description: 'Streaming decode for Windows NVIDIA proxy generation',
  },

  // === Jan 6, 2026 ===
  {
    date: '2026-01-06',
    type: 'new',
    title: '37 Blend Modes',
    description: 'All After Effects blend modes now available',
  },
  {
    date: '2026-01-06',
    type: 'new',
    title: 'Keyframe Animation System',
    description: 'Animate any clip property with keyframes',
  },
  {
    date: '2026-01-06',
    type: 'new',
    title: 'Composition Tabs',
    description: 'Switch between compositions like After Effects',
  },
  {
    date: '2026-01-06',
    type: 'new',
    title: 'Clip Cutter',
    description: 'Press C to cut clips at playhead',
  },
  {
    date: '2026-01-06',
    type: 'new',
    title: 'File System Access',
    description: 'Save proxies and projects to real folders',
  },

  // === Jan 5, 2026 ===
  {
    date: '2026-01-05',
    type: 'new',
    title: 'GPU Proxy Generation',
    description: 'Multi-core accelerated proxy creation',
  },
  {
    date: '2026-01-05',
    type: 'improve',
    title: 'Instant Clip Drop',
    description: 'Clips appear immediately, thumbnails generate in background',
  },

  // === Jan 4, 2026 ===
  {
    date: '2026-01-04',
    type: 'new',
    title: 'RAM Preview',
    description: 'Cache frames to GPU for instant scrubbing like After Effects',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Nested Compositions',
    description: 'Create and playback nested comps with thumbnails',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'IndexedDB Persistence',
    description: 'Projects auto-save locally in your browser',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Global Undo/Redo',
    description: 'Ctrl+Z / Ctrl+Shift+Z for all actions',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Export Panel',
    description: 'In/Out markers and render settings',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Solo & Mute Tracks',
    description: 'Isolate or mute video/audio tracks',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Loop Playback',
    description: 'Toggle loop mode and stop at out point',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Clip Snapping',
    description: 'Smart collision detection when moving clips',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Precision Sliders',
    description: 'Fine control for transform and effect values',
  },
  {
    date: '2026-01-04',
    type: 'new',
    title: 'Audio Track Support',
    description: 'Dedicated audio tracks with playback sync',
  },
  {
    date: '2026-01-04',
    type: 'fix',
    title: 'Blend Modes Working',
    description: 'Fixed u32 vs float mismatch in shader',
  },
  {
    date: '2026-01-04',
    type: 'improve',
    title: 'Frame Caching',
    description: 'Smooth video scrubbing with texture cache',
  },
];

// Group changes by time period
export function getGroupedChangelog(): TimeGroupedChanges[] {
  const groups = new Map<string, { sortOrder: number; dateRange: string; changes: ChangeEntry[] }>();

  for (const entry of RAW_CHANGELOG) {
    const date = new Date(entry.date);
    const { label, sortOrder } = getTimeLabel(date);

    if (!groups.has(label)) {
      // Format date range
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      groups.set(label, { sortOrder, dateRange: dateStr, changes: [] });
    }

    const group = groups.get(label)!;
    group.changes.push({
      type: entry.type,
      title: entry.title,
      description: entry.description,
      section: entry.section,
      commit: entry.commit,
    });

    // Update date range if needed
    const currentDate = new Date(entry.date);
    const dateStr = currentDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    if (!group.dateRange.includes('-') && dateStr !== group.dateRange) {
      // Create a range
      const [firstPart] = group.dateRange.split(' ');
      const [, secondDay] = dateStr.split(' ');
      if (firstPart === dateStr.split(' ')[0]) {
        // Same month
        group.dateRange = `${group.dateRange.split(' ')[0]} ${secondDay}-${group.dateRange.split(' ')[1]}`;
      }
    }
  }

  // Sort groups by sortOrder and return
  return Array.from(groups.entries())
    .map(([label, data]) => ({
      label,
      dateRange: data.dateRange,
      changes: data.changes,
    }))
    .sort((a, b) => {
      const orderA = groups.get(a.label)?.sortOrder ?? 99;
      const orderB = groups.get(b.label)?.sortOrder ?? 99;
      return orderA - orderB;
    });
}

// Legacy interface for backward compatibility
export interface ChangelogEntry {
  version: string;
  date: string;
  changes: {
    type: 'new' | 'fix' | 'improve';
    description: string;
  }[];
}

// Legacy changelog export (keeping for backward compatibility)
export const CHANGELOG: ChangelogEntry[] = [
  {
    version: '1.0.9',
    date: '2026-01-20',
    changes: RAW_CHANGELOG.filter(c => c.date === '2026-01-20').map(c => ({
      type: c.type,
      description: c.title,
    })),
  },
];

// Known issues and bugs - shown in What's New dialog
// Remove items when fixed
export const KNOWN_ISSUES: string[] = [
  'YouTube download requires Native Helper with yt-dlp installed',
  'Audio waveforms may not display for some video formats',
  'Very long videos (>2 hours) may cause performance issues',
];
