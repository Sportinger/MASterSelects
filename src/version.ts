// App version - INCREMENT ON EVERY COMMIT!
// Format: MAJOR.MINOR.PATCH
// Increment PATCH (0.0.X) for each commit
export const APP_VERSION = '1.0.9';

// Changelog entry type
export interface ChangelogEntry {
  version: string;
  date: string;
  changes: {
    type: 'new' | 'fix' | 'improve';
    description: string;
  }[];
}

// Version changelog - add new entries at the TOP
export const CHANGELOG: ChangelogEntry[] = [
  {
    version: '1.0.9',
    date: '2026-01-20',
    changes: [
      { type: 'new', description: 'Audio Master Clock - playhead follows audio for perfect sync (like Premiere/Resolve)' },
      { type: 'new', description: 'Varispeed audio scrubbing - continuous playback with speed adjustment' },
      { type: 'new', description: 'Multi-select in relink dialog for batch file relinking' },
      { type: 'new', description: 'Desktop mode option for mobile devices' },
      { type: 'fix', description: 'MOV/MXF import now works with uppercase extensions (.MOV, .MXF)' },
      { type: 'fix', description: 'Support for all video containers (TS, MTS, 3GP, etc.) with H.264/H.265' },
      { type: 'fix', description: 'Export frame sync - proper H.264 avcC config and sequential decoding' },
      { type: 'fix', description: 'Auto-open folder picker after single file relink' },
      { type: 'improve', description: 'Layer caching for better pause/scrubbing performance' },
      { type: 'improve', description: 'WebCodecs export optimization with parallel clip loading' },
      { type: 'improve', description: 'Varispeed scrubbing works for all video clips (not just proxies)' },
    ],
  },
  {
    version: '1.0.8',
    date: '2026-01-13',
    changes: [
      { type: 'new', description: 'YouTube video download via yt-dlp (requires Native Helper)' },
      { type: 'new', description: 'Auto-add video when pasting YouTube URLs' },
      { type: 'new', description: 'Download button on YouTube video thumbnails' },
      { type: 'new', description: 'NativeDecoder integration for ProRes/DNxHD playback' },
      { type: 'fix', description: 'Export progress bar now shows accurate progress' },
      { type: 'improve', description: 'Better error handling for video file imports' },
    ],
  },
  {
    version: '1.0.7',
    date: '2026-01-13',
    changes: [
      { type: 'new', description: 'Native Helper app for hardware-accelerated video codecs' },
      { type: 'fix', description: 'Frame-accurate seeking for FFmpeg export' },
      { type: 'new', description: 'Audio support in FFmpeg export with progress bar' },
      { type: 'new', description: 'Custom FFmpeg WASM build with professional codecs' },
    ],
  },
  {
    version: '1.0.6',
    date: '2026-01-10',
    changes: [
      { type: 'improve', description: 'YouTubePanel component updates' },
      { type: 'new', description: 'Feature documentation system' },
    ],
  },
];

// Known issues and bugs - shown in What's New dialog
// Remove items when fixed
export const KNOWN_ISSUES: string[] = [
  'YouTube download requires Native Helper with yt-dlp installed',
  'Audio waveforms may not display for some video formats',
  'Very long videos (>2 hours) may cause performance issues',
];
