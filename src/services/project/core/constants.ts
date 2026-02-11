// Project folder structure constants

export const PROJECT_FOLDERS = {
  RAW: 'Raw',
  PROXY: 'Proxy',
  ANALYSIS: 'Analysis',
  TRANSCRIPTS: 'Transcripts',
  CACHE: 'Cache',
  CACHE_THUMBNAILS: 'Cache/thumbnails',
  CACHE_WAVEFORMS: 'Cache/waveforms',
  RENDERS: 'Renders',
  BACKUPS: 'Backups',
  DOWNLOADS: 'Downloads',
} as const;

export type ProjectFolderKey = keyof typeof PROJECT_FOLDERS;

export const MAX_BACKUPS = 20;

// All folders to create when initializing a project
export const PROJECT_FOLDER_PATHS = [
  PROJECT_FOLDERS.RAW,
  PROJECT_FOLDERS.PROXY,
  PROJECT_FOLDERS.ANALYSIS,
  PROJECT_FOLDERS.TRANSCRIPTS,
  PROJECT_FOLDERS.CACHE,
  PROJECT_FOLDERS.CACHE_THUMBNAILS,
  PROJECT_FOLDERS.CACHE_WAVEFORMS,
  PROJECT_FOLDERS.RENDERS,
  PROJECT_FOLDERS.BACKUPS,
  PROJECT_FOLDERS.DOWNLOADS,
] as const;
