// App version - INCREMENT ON EVERY COMMIT!
// Format: MAJOR.MINOR.PATCH
// Increment PATCH (0.0.X) for each commit
export const APP_VERSION = '1.2.3';

// Build/Platform notice shown at top of changelog (set to null to hide)
export const BUILD_NOTICE: {
  type: 'info' | 'warning' | 'success';
  title: string;
  message: string;
} | null = null;

// Change entry type
export interface ChangeEntry {
  type: 'new' | 'fix' | 'improve' | 'refactor';
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
  // === Feb 9, 2026 - Tutorial System, SAM2 AI Segmentation, Composition Resolution ===
  {
    date: '2026-02-09',
    type: 'new',
    title: 'Interactive Tutorial with Clippy',
    description: 'Two-part guided tutorial: Part 1 introduces panels with spotlight overlay, Part 2 deep-dives into timeline features. Animated Clippy mascot with intro/outro animations',
    commit: '9b97b62',
  },
  {
    date: '2026-02-09',
    type: 'new',
    title: 'Tutorial Welcome Screen',
    description: 'Choose your NLE background (Premiere Pro, DaVinci Resolve, Final Cut Pro, After Effects, Beginner) at tutorial start — stored for future personalization',
    commit: '78397d3',
  },
  {
    date: '2026-02-09',
    type: 'new',
    title: 'SAM 2 AI Segmentation',
    description: 'AI-powered object segmentation using Segment Anything Model 2 — click to select objects in the preview with GPU-accelerated inference',
    commit: 'abd2c9a',
  },
  {
    date: '2026-02-09',
    type: 'new',
    title: 'Vitest Test Suite',
    description: '182 unit and store tests covering timeline, media, history and utility modules',
    commit: 'f74f65a',
  },
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'Composition-Driven Resolution',
    description: 'Active composition resolution now drives the render pipeline instead of a global setting — supports per-comp resolution',
    commit: '80bece4',
  },
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'Native Pixel Scale for Clips',
    description: 'Newly added clips and composition resize auto-adjust scale for pixel-accurate content display',
    commit: '1f47ce8',
  },
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'Reorganized Menus',
    description: 'Scopes moved to Panels submenu in View, AI panels grouped as submenu, tutorials accessible from Info menu',
    commit: '2122520',
  },
  {
    date: '2026-02-09',
    type: 'fix',
    title: 'Clip Overlap & Audio Desync',
    description: 'Fixed clips overlapping and audio desyncing when moving clips on the same track',
    commit: 'c5c29f5',
  },
  {
    date: '2026-02-09',
    type: 'fix',
    title: 'SAM2 Model Loading',
    description: 'Use ORT-optimized encoder from webgpu-sam2 CDN with correct tensor names and mask overlay alignment',
    commit: '7c28463',
  },
  {
    date: '2026-02-09',
    type: 'fix',
    title: 'Wrong Track Type Drag Prevention',
    description: 'Clips can no longer be dragged to incompatible track types in preview',
    commit: '03aa49a',
  },

  // === Feb 9, 2026 - Comp Switch Animations, Label Colors, Timeline Polish ===
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'Composition Switch Animations',
    description: 'Smooth 200ms crossfade with per-track slide-in animation when switching compositions',
    commit: '4266604',
  },
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'Staggered Clip Entrance',
    description: 'Clips animate in sequentially with 20ms delay per clip for a cascading entrance effect',
    commit: 'c3efc3e',
  },
  {
    date: '2026-02-09',
    type: 'new',
    title: 'Clip Label Colors',
    description: 'Timeline clips inherit label color from their media file or composition in the media panel',
    commit: '8ead363',
  },
  {
    date: '2026-02-09',
    type: 'new',
    title: 'Sortable Media Panel Columns',
    description: 'Label color column as proper sortable column with clickable column headers',
    commit: '865c45d',
  },
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'Audio Tracks Default Expanded',
    description: 'Audio tracks always show expand arrow and start expanded by default',
    commit: 'e194550',
  },
  {
    date: '2026-02-09',
    type: 'improve',
    title: 'NativeHelper Auto-Connect',
    description: 'Auto-check connection status when NativeHelper dialog opens, updated dialog to focus on YouTube downloading',
    commit: '1441015',
  },
  {
    date: '2026-02-09',
    type: 'fix',
    title: 'Timeline Duration Sync',
    description: 'Timeline duration edits now correctly sync and persist to the active composition',
    commit: '6b48cf8',
  },
  {
    date: '2026-02-09',
    type: 'refactor',
    title: 'Disable Marquee Selection & Pick Whip',
    description: 'Removed marquee/lasso selection and pick whip parenting from timeline',
    commit: '6a0b69b',
  },

  // === Feb 8, 2026 - Scopes, Curve Editor, Timeline UX, Mask Performance ===
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Video Scopes Panel',
    description: 'GPU-accelerated Histogram, Vectorscope and Waveform monitor — zero readPixels overhead with DaVinci Resolve-style rendering',
    commit: '8c735e8',
  },
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Waveform Monitor',
    description: 'DaVinci-style waveform with smooth phosphor glow traces, sub-pixel distribution and bilinear sampling',
    commit: '1cf164e',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Keyframe Curve Editor',
    description: 'Auto-scale Y-axis to fit curve tightly, Shift+wheel to resize height, fix scroll offset and overlapping rows',
    commit: '4ccbcbc',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Single Curve Editor',
    description: 'Only one curve editor open at a time to prevent UI clutter',
    commit: '48af4ac',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Keyframe Toggle Off',
    description: 'Toggling keyframes off now saves the current value and deletes all keyframes cleanly',
    commit: '0e19f8b',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Timeline Vertical Scroll Snapping',
    description: 'Vertical scrolling in the timeline snaps to track boundaries — each scroll step moves exactly one layer',
    commit: 'a62e57a',
  },
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Video/Audio Track Separator',
    description: 'Green divider line between video and audio tracks for clearer visual structure',
    commit: '3f16953',
  },
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Mask Edge Dragging',
    description: 'Drag a line segment between two mask vertices to move both at once — works alongside vertex and whole-mask dragging',
    commit: '3f16953',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Mask Drag Performance',
    description: 'Skip history snapshots during mask dragging, GPU texture updates at 30fps instead of 10fps, targeted cache invalidation',
    commit: '867d843',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Compact Settings Dialog',
    description: 'Consolidated all API keys into settings and improved layout',
    commit: '716747f',
  },
  {
    date: '2026-02-08',
    type: 'fix',
    title: 'View Toggle Checkboxes',
    description: 'View dropdown checkboxes now update visually on click and thumbnails/waveforms toggles actually hide content',
    commit: '01715bf',
  },
  {
    date: '2026-02-08',
    type: 'fix',
    title: 'WGSL Reserved Keyword',
    description: 'Renamed WGSL reserved keyword "ref" to "rv" in scope shaders',
    commit: 'cf05b82',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Persist View Toggles',
    description: 'View toggle states (thumbnails, waveforms, scopes) saved in project file',
    commit: '49dc4cd',
  },
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Separate Scope Panels',
    description: 'Histogram, Waveform and Vectorscope are now 3 independent panels with RGB/R/G/B/Luma view mode buttons and IRE legend',
    commit: '36b190f',
  },
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Keyframe Copy/Paste',
    description: 'Ctrl+C/V with selected keyframes copies only keyframes — paste at playhead position on the selected clip',
    commit: '604e20f',
  },
  {
    date: '2026-02-08',
    type: 'new',
    title: 'Keyframe Tick Marks on Clips',
    description: 'Small amber diamond markers at the bottom of clips show keyframe positions without expanding tracks',
    commit: '9a65a52',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Exponential Timeline Zoom',
    description: 'Alt+Scroll zoom now uses exponential scaling (8% per step) — consistent feel at all zoom levels',
    commit: '1642bb9',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Cross-Track Clip Movement',
    description: 'Smart overlap prevention on track changes — find free track or create new one. 100px vertical resistance prevents accidental moves',
    commit: '6a47f4e',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'Track Height: Smooth & Compact',
    description: 'Continuous scrolling for track height (no fixed steps), minimum 20px for ultra-compact view with single line of text',
    commit: '080b9fc',
  },
  {
    date: '2026-02-08',
    type: 'improve',
    title: 'High-Res Vectorscope',
    description: 'Higher resolution vectorscope with smooth histogram rendering and fixed white blob issue',
    commit: 'd6cae47',
  },
  {
    date: '2026-02-08',
    type: 'fix',
    title: 'Ctrl+Z Crash Fix',
    description: 'Fixed undo crash caused by undefined layers in history snapshots',
    commit: '7559e0b',
  },
  {
    date: '2026-02-08',
    type: 'fix',
    title: 'Ruler Click Position',
    description: 'Fixed ruler click using wrong reference element for position calculation',
    commit: '723b76c',
  },
  {
    date: '2026-02-08',
    type: 'fix',
    title: 'Histogram Peak Clipping',
    description: 'Increased normalization headroom to prevent histogram peaks from being cut off',
    commit: 'cc1bf39',
  },

  // === Feb 7, 2026 - Linked Clips, Proxy Rewrite, Split Fix, MOV Import ===
  {
    date: '2026-02-07',
    type: 'new',
    title: 'Linked Clip Selection',
    description: 'Click a linked video or audio clip to select both - Shift+click for independent selection. Properties panel shows the clicked clip.',
    commit: 'e51ce42',
  },
  {
    date: '2026-02-07',
    type: 'new',
    title: 'Proxy Resume from Disk',
    description: 'Proxy generation resumes from disk after interruption instead of starting over',
    commit: 'bd4259d',
  },
  {
    date: '2026-02-07',
    type: 'improve',
    title: 'Proxy System Rewrite',
    description: 'New proxy pipeline: WebCodecs decode with parallel JPEG encoding for faster proxy generation',
    commit: '6224538',
  },
  {
    date: '2026-02-07',
    type: 'improve',
    title: 'Instant Media Import',
    description: 'Media files appear instantly in timeline - faster MOV duration extraction via MP4Box container parsing',
    commit: '88f3344',
  },
  {
    date: '2026-02-07',
    type: 'improve',
    title: 'Playback Performance',
    description: 'Decoupled playhead position from clip components to prevent unnecessary re-renders during playback',
    commit: 'f9f996c',
  },
  {
    date: '2026-02-07',
    type: 'fix',
    title: 'Split Clips Fully Independent',
    description: 'Cut/split clips now deep-clone transform, effects, masks and text properties instead of sharing them',
    commit: 'ef8d208',
  },
  {
    date: '2026-02-07',
    type: 'fix',
    title: 'First-Time Save Data Loss',
    description: 'Saving a project for the first time no longer loses all edits - folder creation no longer wipes store data',
    commit: 'ef62d34',
  },
  {
    date: '2026-02-07',
    type: 'fix',
    title: 'Transparency Grid in GPU Shader',
    description: 'Checkerboard transparency grid now renders directly in the GPU shader with correct sizing',
    commit: '9c6012a',
  },
  {
    date: '2026-02-07',
    type: 'fix',
    title: 'Undo/Redo Reliability',
    description: 'Flush pending captures before undo/redo, shallow equality checks, improved batch safety',
    commit: '9e8c220',
  },
  {
    date: '2026-02-07',
    type: 'fix',
    title: 'Keyframe Drag Fix',
    description: 'Clicking a new keyframe no longer drags the previously selected one',
    commit: 'e87fd67',
  },
  {
    date: '2026-02-07',
    type: 'fix',
    title: 'Blend Mode German Keyboard',
    description: 'Blend mode cycling now works with direct + key on German keyboard layout',
    commit: '173fc8d',
  },

  // === Feb 6, 2026 - Visual Redesign, Solid Clips, Properties Panel ===
  {
    date: '2026-02-06',
    type: 'new',
    title: 'Solid Color Clips',
    description: 'Create solid color clips from the Media Panel - with color picker, comp dimensions, and colored timeline bars',
    commit: '248996f',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'After Effects Visual Redesign',
    description: 'Complete UI overhaul - darker color palette (#0f0f0f base), refined typography and proportions',
    commit: '290d15e',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'AE-Style Media Panel',
    description: 'File type icons instead of thumbnails, color label column, folder disclosure triangles',
    commit: 'd6c72a2',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'AE-Style Properties Panel',
    description: 'Unnormalized values (0-100 instead of 0-1), compact inline X/Y/Z, label-value alignment',
    commit: '5902d6a',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Live Text Preview in Canvas',
    description: 'Text changes now update live in the composition canvas + dynamic font weight dropdown',
    commit: '79020a7',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'Inline GPU Effects',
    description: 'Brightness, contrast, saturation and invert run inside composite shader - no extra render passes',
    commit: '58178e8',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Undo/Redo History System',
    description: 'Fixed race conditions, missing state captures, and batch grouping in the history system',
    commit: '804ce7f',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Audio Detection for Camera MOV',
    description: 'Fixed audio tracks not detected in Canon DK7A and other camera MOV files',
    commit: '540a3e6',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Persist Text & Solid Items',
    description: 'Text items and solid items in the media panel now survive page reloads',
    commit: '345a6fe',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Proxy System Stability',
    description: 'Fixed DPB deadlock on NVIDIA GPUs, slow start, and stuck at 100% issues',
    commit: '345c3f3',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Panel Resize Handles',
    description: 'Wider grab area, centered on visual edge, only the dragged handle highlights',
    commit: '22858cd',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Clip Loading UX',
    description: 'Instant placeholder on drop, hidden loading audio clips, overflow containment',
    commit: 'cf465aa',
  },
  {
    date: '2026-02-06',
    type: 'refactor',
    title: 'Timeline Store Split',
    description: 'Extracted positioning and serialization utils for cleaner architecture',
    commit: '07f5ffb',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'Timeline Selector Bundling',
    description: 'Reduced 29 individual store subscriptions to 6 with useShallow',
    commit: '96ff2aa',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'Lazy-Load Panels',
    description: 'Split heavy panel dependencies for 19% smaller main chunk',
    commit: 'eb8b569',
  },
  {
    date: '2026-02-06',
    type: 'improve',
    title: 'RAF-Debounce Layer Sync',
    description: 'Smoother scrubbing by debouncing layer synchronization to animation frames',
    commit: 'e352aa5',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Memory Leak Prevention',
    description: 'Fixed RenderLoop recreating on every play/pause + added RAM preview cache limit',
    commit: 'dc62e03',
  },
  {
    date: '2026-02-06',
    type: 'fix',
    title: 'Project Rename',
    description: 'Fixed rename failing with "Folder already exists" error',
    commit: '2a351e4',
  },

  // === Feb 4, 2026 - React Optimization & RenderLoop Fixes ===
  {
    date: '2026-02-04',
    type: 'refactor',
    title: 'PropertiesPanel Code Splitting',
    description: 'Split 61K file into 8 lazy-loaded modules for faster initial load',
    commit: 'dbd9a0f',
  },
  {
    date: '2026-02-04',
    type: 'refactor',
    title: 'Store Subscription Optimization',
    description: 'Fixed 12+ components subscribing to entire store - now use getState() for actions',
    commit: 'a739556',
  },
  {
    date: '2026-02-04',
    type: 'refactor',
    title: 'React Performance Optimization',
    description: 'Zustand selectors and lazy loading for reduced re-renders',
    commit: '0352751',
  },
  {
    date: '2026-02-04',
    type: 'refactor',
    title: 'Extract useMarkerDrag Hook',
    description: 'Marker drag logic extracted into reusable custom hook',
    commit: '828b090',
  },
  {
    date: '2026-02-04',
    type: 'refactor',
    title: 'Stabilize TimelineControls',
    description: 'useCallback for callback props to prevent unnecessary re-renders',
    commit: 'e1752ed',
  },
  {
    date: '2026-02-04',
    type: 'improve',
    title: 'Draggable Settings Dialog',
    description: 'Settings dialog can now be dragged without dark overlay',
    commit: '977aefc',
  },
  {
    date: '2026-02-04',
    type: 'fix',
    title: 'Infinite Loop in Actions',
    description: 'Use getState() for store actions instead of selectors to prevent loops',
    commit: 'f839fb2',
  },
  {
    date: '2026-02-04',
    type: 'fix',
    title: 'Proxy Generation at 103%',
    description: 'Clamped proxy progress and ensured completion callback fires',
    commit: '25cfbf4',
  },
  {
    date: '2026-02-04',
    type: 'fix',
    title: 'Audio Clip Fades',
    description: 'Audio fade handles now control Volume instead of Opacity',
    commit: '63d2214',
  },
  {
    date: '2026-02-04',
    type: 'fix',
    title: 'RAM Preview Playback',
    description: 'Fixed video/audio not playing when RAM Preview cache is used',
    commit: 'b9cba69',
  },
  {
    date: '2026-02-04',
    type: 'fix',
    title: 'RenderLoop Memory Leak',
    description: 'Fixed RenderLoop being recreated on every play/pause causing memory leaks',
    commit: 'dab0e6d',
  },

  // === Feb 3, 2026 - Transitions, Multi-Select, FCP XML & More ===
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Transitions System',
    description: 'Crossfade transitions between clips with GPU-accelerated rendering',
    commit: '455a99d',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Transitions Panel',
    description: 'Modular panel with drag-drop support for applying transitions',
    commit: 'a98c2ad',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'JKL Playback Shortcuts',
    description: 'Industry-standard J/K/L keyboard shortcuts for playback control',
    commit: 'c0f882f',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Transform Handles in Edit Mode',
    description: 'Corner and edge handles for scaling with Shift for aspect ratio lock',
    commit: '633b842',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Proxy Cache Indicator',
    description: 'Yellow indicator on timeline ruler shows cached proxy frames',
    commit: 'fb19843',
  },
  {
    date: '2026-02-03',
    type: 'new',
    title: 'Manual Video Warmup',
    description: 'Cache button for preloading proxy frames before playback',
    commit: '0f1a902',
  },
  {
    date: '2026-02-03',
    type: 'improve',
    title: 'Settings Dialog Redesign',
    description: 'After Effects-style sidebar navigation with categorized settings',
    commit: '1c8cbeb',
  },
  {
    date: '2026-02-03',
    type: 'improve',
    title: 'Faster RAM Preview',
    description: 'Use WebCodecsPlayer for faster RAM Preview frame generation',
    commit: '5e618fc',
  },
  {
    date: '2026-02-03',
    type: 'improve',
    title: 'Reverse Playback',
    description: 'Improved reverse playback support for H.264 videos',
    commit: 'b19a5cb',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'RAM Preview Nested Compositions',
    description: 'RAM Preview now correctly caches nested composition frames',
    commit: '52d922d',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Reset Playback Speed on Pause',
    description: 'Playback speed resets to 1x when pausing instead of staying at JKL speed',
    commit: 'cb33627',
  },
  {
    date: '2026-02-03',
    type: 'fix',
    title: 'Audio Effect GPU Warnings',
    description: 'Fixed audio effect warnings in GPU rendering pipeline',
    commit: 'a6ff1dc',
  },

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
  {
    date: '2026-02-02',
    type: 'new',
    title: 'MIT License',
    description: 'Project released under MIT open-source license',
    commit: '7f62c8d',
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
    type: 'new',
    title: 'Fade Curve Bezier Display',
    description: 'Real-time bezier curve visualization for opacity fades directly on timeline clips',
    commit: 'db70328',
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
    type: 'refactor',
    title: 'WebGPUEngine Refactor',
    description: 'Split into focused modules - 57% smaller core file',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
    title: 'Timeline Component Refactor',
    description: 'Extract hooks and utilities - 2109 → 1323 lines of code',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
    title: 'ClipSlice Refactor',
    description: 'Modular clip and helper modules - 66% code reduction',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
    title: 'FrameExporter Refactor',
    description: 'Split 1510-line file into 8 focused modules',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
    title: 'ProjectFileService Refactor',
    description: 'Modular architecture with clean separation of concerns',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
    title: 'AI Tools Refactor',
    description: 'Split monolithic aiTools.ts into modular architecture',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
    title: 'Remove VJ/Mixer Mode',
    description: 'Simplified codebase by removing unused VJ mode entirely',
  },
  {
    date: '2026-01-25',
    type: 'refactor',
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
  {
    date: '2026-01-25',
    type: 'new',
    title: 'Auto-Copy Media to Raw Folder',
    description: 'Imported media files automatically copied to project Raw folder for portability',
    commit: 'd42f45c',
  },
  {
    date: '2026-01-25',
    type: 'new',
    title: 'Auto-Relink from Raw Folder',
    description: 'Missing media files automatically relinked from Raw folder on project load',
    commit: '0e3d7ed',
  },
  {
    date: '2026-01-25',
    type: 'new',
    title: 'Cut Tool Snapping',
    description: 'Cut tool snaps to playhead, clip edges and markers - hold Alt to disable',
    commit: '1b04016',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Cut Indicator Across Linked Clips',
    description: 'Cut line extends across linked video+audio clips for visual clarity',
    commit: '4e19381',
  },
  {
    date: '2026-01-25',
    type: 'improve',
    title: 'Zero-Copy Export Path',
    description: 'Export uses OffscreenCanvas → VideoFrame for faster GPU-to-encoder transfer',
    commit: '2f0c5aa',
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
    type: 'refactor',
    title: 'Removed VJ/Mixer Mode',
    description: 'Simplified codebase by removing unused VJ mode',
  },
  {
    date: '2026-01-21',
    type: 'refactor',
    title: 'Logger Service Migration',
    description: 'Migrated codebase to centralized Logger service',
    commit: '847a386',
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
    type: 'refactor',
    title: 'WebCodecsPlayer Export Simplification',
    description: 'Simplified WebCodecsPlayer export mode for cleaner code',
    commit: 'a1d193f',
  },
  {
    date: '2026-01-20',
    type: 'refactor',
    title: 'View Toggles Consolidation',
    description: 'Consolidated view toggles into single dropdown button',
    commit: 'ef9e3c0',
  },
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
  {
    date: '2026-01-20',
    type: 'new',
    title: 'Parallel Video Decoding',
    description: 'Multi-clip parallel decode for dramatically faster exports with nested compositions',
    commit: '083591b',
  },
  {
    date: '2026-01-20',
    type: 'new',
    title: 'YouTube Quality Selection',
    description: 'Choose video quality and codec (H.264/VP9/AV1) when downloading YouTube videos',
    commit: '978308e',
  },
  {
    date: '2026-01-20',
    type: 'new',
    title: 'macOS Menubar Helper',
    description: 'Native macOS menubar app with tray icon for helper management',
    commit: 'aac6c45',
  },
  {
    date: '2026-01-20',
    type: 'improve',
    title: 'Mask Editing Performance',
    description: 'Throttled mask texture updates to 10fps during drag for smooth vertex editing',
    commit: '4766f3a',
  },

  // === Jan 23, 2026 - YouTube & Native Helper ===
  {
    date: '2026-01-23',
    type: 'new',
    title: 'GPU Vendor Display',
    description: 'Show GPU vendor (Nvidia, AMD, Intel) in app header for quick hardware identification',
    commit: '1675636',
  },
  {
    date: '2026-01-23',
    type: 'new',
    title: 'YouTube Search Persistence',
    description: 'YouTube searches saved per project with downloads stored in project YT folder',
    commit: '7a6be26',
  },
  {
    date: '2026-01-23',
    type: 'new',
    title: 'YouTube Format Selection',
    description: 'Choose video quality and format before downloading YouTube videos',
    commit: '1af642a',
  },
  {
    date: '2026-01-23',
    type: 'new',
    title: 'Lite Helper for Windows',
    description: 'Lightweight YouTube-only helper without FFmpeg dependency',
    commit: '374a147',
  },
  {
    date: '2026-01-23',
    type: 'fix',
    title: 'WebGPU Device Loss Recovery',
    description: 'Graceful handling and automatic recovery from GPU device loss',
    commit: '13cba3d',
  },
  {
    date: '2026-01-23',
    type: 'fix',
    title: 'GPU Memory Exhaustion',
    description: 'Guard against Vulkan OOM with smaller initial textures and delayed allocation',
    commit: '5d68416',
  },

  // === Jan 16, 2026 ===
  {
    date: '2026-01-16',
    type: 'refactor',
    title: 'Helpers Platform Reorganization',
    description: 'Reorganized native helpers by platform (win, mac, linux)',
    commit: '6a51335',
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
    type: 'new' | 'fix' | 'improve' | 'refactor';
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
