// RelinkDialog - Dialog to relink missing media files
// Shows list of missing files, allows searching folders, updates status

import { useState, useCallback, useEffect } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import { fileSystemService } from '../../services/fileSystemService';
import { projectDB } from '../../services/projectDB';

interface RelinkDialogProps {
  onClose: () => void;
}

interface FileStatus {
  id: string;
  name: string;
  filePath?: string;
  status: 'missing' | 'found' | 'searching';
  newFile?: File;
  newHandle?: FileSystemFileHandle;
}

export function RelinkDialog({ onClose }: RelinkDialogProps) {
  const { files } = useMediaStore();
  const [fileStatuses, setFileStatuses] = useState<FileStatus[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchedFolders, setSearchedFolders] = useState<string[]>([]);

  // Initialize file statuses
  useEffect(() => {
    const missingFiles = files.filter(f => !f.file);
    setFileStatuses(missingFiles.map(f => ({
      id: f.id,
      name: f.name,
      filePath: f.filePath,
      status: 'missing',
    })));
  }, [files]);

  // Scan a folder for missing files
  const scanFolder = useCallback(async (dirHandle: FileSystemDirectoryHandle) => {
    setIsSearching(true);

    // Collect all files from directory recursively
    const foundFiles = new Map<string, FileSystemFileHandle>();

    const scanDirectory = async (dir: FileSystemDirectoryHandle) => {
      try {
        for await (const entry of (dir as any).values()) {
          if (entry.kind === 'file') {
            const fileName = entry.name.toLowerCase();
            foundFiles.set(fileName, entry);
          } else if (entry.kind === 'directory') {
            await scanDirectory(entry);
          }
        }
      } catch (e) {
        console.warn('[RelinkDialog] Error scanning directory:', e);
      }
    };

    await scanDirectory(dirHandle);
    console.log('[RelinkDialog] Found', foundFiles.size, 'files in', dirHandle.name);

    // Match missing files
    setFileStatuses(prev => {
      const updated = [...prev];
      for (const status of updated) {
        if (status.status === 'missing') {
          const searchName = status.name.toLowerCase();
          const handle = foundFiles.get(searchName);

          if (handle) {
            status.status = 'found';
            status.newHandle = handle;
            // Get the file async
            handle.getFile().then(file => {
              setFileStatuses(curr => curr.map(s =>
                s.id === status.id ? { ...s, newFile: file } : s
              ));
            });
          }
        }
      }
      return updated;
    });

    setSearchedFolders(prev => [...prev, dirHandle.name]);
    setIsSearching(false);
  }, []);

  // Handle browse button
  const handleBrowse = useCallback(async () => {
    try {
      const dirHandle = await (window as any).showDirectoryPicker({
        mode: 'read',
        startIn: 'videos',
      });

      if (dirHandle) {
        await scanFolder(dirHandle);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        console.error('[RelinkDialog] Browse error:', e);
      }
    }
  }, [scanFolder]);

  // Handle picking individual file - allows multiple selection to relink several at once
  const handlePickFile = useCallback(async (fileStatus: FileStatus) => {
    // Check how many files are still missing
    const missingFiles = fileStatuses.filter(s => s.status === 'missing');
    const allowMultiple = missingFiles.length > 1;

    try {
      const handles = await (window as any).showOpenFilePicker({
        multiple: allowMultiple, // Allow multiple selection if there are multiple missing files
        types: [{
          description: allowMultiple
            ? `Select missing files (${missingFiles.length} missing)`
            : `Locate: ${fileStatus.name}`,
          accept: {
            'video/*': [],
            'audio/*': [],
            'image/*': [],
          },
        }],
      });

      if (handles && handles.length > 0) {
        // Build a map of selected files by name (lowercase for matching)
        const selectedFiles = new Map<string, { file: File; handle: FileSystemFileHandle }>();

        for (const handle of handles) {
          const file = await handle.getFile();
          selectedFiles.set(file.name.toLowerCase(), { file, handle });
        }

        console.log(`[RelinkDialog] User selected ${selectedFiles.size} file(s)`);

        // Match selected files against missing files
        setFileStatuses(prev => prev.map(status => {
          if (status.status === 'missing') {
            const match = selectedFiles.get(status.name.toLowerCase());
            if (match) {
              console.log(`[RelinkDialog] Matched: ${status.name}`);
              return {
                ...status,
                status: 'found',
                newFile: match.file,
                newHandle: match.handle,
              };
            }
          }
          return status;
        }));

        // If there are still missing files after selection, offer to scan the folder
        const stillMissing = fileStatuses.filter(s =>
          s.status === 'missing' && !selectedFiles.has(s.name.toLowerCase())
        );

        if (stillMissing.length > 0 && handles.length > 0) {
          // Automatically open folder picker starting from the selected file's location
          try {
            const dirHandle = await (window as any).showDirectoryPicker({
              mode: 'read',
              startIn: handles[0], // Start in same folder as selected file
            });
            if (dirHandle) {
              console.log('[RelinkDialog] Scanning folder for remaining files...');
              await scanFolder(dirHandle);
            }
          } catch (e: any) {
            // User cancelled - that's fine, we still have the manually selected files
            if (e.name !== 'AbortError') {
              console.log('[RelinkDialog] Folder access declined, using manually selected files only');
            }
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        console.error('[RelinkDialog] Pick file error:', e);
      }
    }
  }, [fileStatuses, scanFolder]);

  // Apply all found files
  const handleApply = useCallback(async () => {
    const { useTimelineStore } = await import('../../stores/timeline');
    const timelineStore = useTimelineStore.getState();

    for (const status of fileStatuses) {
      if (status.status === 'found' && status.newFile && status.newHandle) {
        const url = URL.createObjectURL(status.newFile);

        // Store handle
        fileSystemService.storeFileHandle(status.id, status.newHandle);
        await projectDB.storeHandle(`media_${status.id}`, status.newHandle);

        // Update media store
        useMediaStore.setState(state => ({
          files: state.files.map(f =>
            f.id === status.id
              ? { ...f, file: status.newFile, url, hasFileHandle: true }
              : f
          ),
        }));

        // Update timeline clips
        const clips = timelineStore.clips.filter(
          c => c.source?.mediaFileId === status.id && c.needsReload
        );

        for (const clip of clips) {
          timelineStore.updateClip(clip.id, {
            file: status.newFile,
            needsReload: false,
            isLoading: true,
          });
        }

        console.log('[RelinkDialog] Applied:', status.name);
      }
    }

    onClose();
  }, [fileStatuses, onClose]);

  const missingCount = fileStatuses.filter(s => s.status === 'missing').length;
  const foundCount = fileStatuses.filter(s => s.status === 'found').length;

  return (
    <div className="welcome-overlay-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="welcome-overlay relink-dialog">
        <h2 className="relink-title">Relink Media</h2>
        <p className="relink-subtitle">
          {missingCount} missing · {foundCount} found
        </p>

        {/* File list */}
        <div className="relink-file-list">
          {fileStatuses.map(status => (
            <div
              key={status.id}
              className={`relink-file-item ${status.status}`}
              onClick={() => status.status === 'missing' && handlePickFile(status)}
            >
              <span className={`relink-status-icon ${status.status}`}>
                {status.status === 'missing' ? '!' : status.status === 'found' ? '✓' : '...'}
              </span>
              <div className="relink-file-info">
                <span className="relink-file-name">{status.name}</span>
                {status.filePath && status.filePath !== status.name && (
                  <span className="relink-file-path">{status.filePath}</span>
                )}
              </div>
              {status.status === 'missing' && (
                <span className="relink-pick-hint">
                  {missingCount > 1 ? 'Click to select files' : 'Click to locate'}
                </span>
              )}
            </div>
          ))}
        </div>

        {/* Searched folders */}
        {searchedFolders.length > 0 && (
          <div className="relink-searched">
            Searched: {searchedFolders.join(', ')}
          </div>
        )}

        {/* Actions */}
        <div className="relink-actions">
          <button
            className="relink-btn relink-btn-secondary"
            onClick={handleBrowse}
            disabled={isSearching}
          >
            {isSearching ? 'Searching...' : 'Search Folder...'}
          </button>
          <div className="relink-actions-right">
            <button className="relink-btn" onClick={onClose}>
              Cancel
            </button>
            <button
              className="relink-btn relink-btn-primary"
              onClick={handleApply}
              disabled={foundCount === 0}
            >
              Apply ({foundCount})
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
