// Bootstrap function to inject Output Manager React root into a popup window
// Since same-origin popups share the JS heap, all stores and engine work directly

import { createRoot } from 'react-dom/client';
import { createElement } from 'react';
import { OutputManager } from './OutputManager';

let managerWindow: Window | null = null;

export function openOutputManager(): void {
  // If already open, focus it
  if (managerWindow && !managerWindow.closed) {
    managerWindow.focus();
    return;
  }

  const win = window.open(
    '',
    'output_manager',
    'width=900,height=600,menubar=no,toolbar=no,location=no,status=no'
  );

  if (!win) {
    console.error('Failed to open Output Manager (popup blocked?)');
    return;
  }

  managerWindow = win;
  win.document.title = 'Output Manager';

  // Inject the main app stylesheet
  const mainStylesheet = document.querySelector('link[rel="stylesheet"]') as HTMLLinkElement | null;
  if (mainStylesheet) {
    const link = win.document.createElement('link');
    link.rel = 'stylesheet';
    link.href = mainStylesheet.href;
    win.document.head.appendChild(link);
  }

  // Inject Output Manager specific styles
  const style = win.document.createElement('style');
  style.textContent = getOutputManagerStyles();
  win.document.head.appendChild(style);

  // Set base styles on body
  win.document.body.style.cssText = 'margin:0;padding:0;background:#0f0f0f;color:#d4d4d4;font-family:system-ui,-apple-system,sans-serif;font-size:13px;overflow:hidden;';

  // Create root element
  const root = win.document.createElement('div');
  root.id = 'output-manager-root';
  root.style.cssText = 'width:100vw;height:100vh;';
  win.document.body.appendChild(root);

  // Mount React
  const reactRoot = createRoot(root);
  reactRoot.render(createElement(OutputManager));

  // Cleanup on close
  win.onbeforeunload = () => {
    reactRoot.unmount();
    managerWindow = null;
  };
}

function getOutputManagerStyles(): string {
  return `
    .om-container {
      display: flex;
      flex-direction: column;
      height: 100vh;
      background: #0f0f0f;
      color: #d4d4d4;
    }
    .om-header {
      display: flex;
      align-items: center;
      padding: 8px 16px;
      background: #161616;
      border-bottom: 1px solid #2a2a2a;
      flex-shrink: 0;
    }
    .om-title {
      margin: 0;
      font-size: 14px;
      font-weight: 600;
      color: #e0e0e0;
    }
    .om-body {
      display: flex;
      flex: 1;
      min-height: 0;
    }
    .om-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #0a0a0a;
      min-width: 0;
    }
    .om-sidebar {
      width: 320px;
      flex-shrink: 0;
      background: #161616;
      border-left: 1px solid #2a2a2a;
      overflow-y: auto;
    }

    /* Target List */
    .om-target-list {
      display: flex;
      flex-direction: column;
      height: 100%;
    }
    .om-target-list-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid #2a2a2a;
    }
    .om-target-list-title {
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #888;
    }
    .om-header-buttons {
      display: flex;
      gap: 4px;
    }
    .om-add-btn {
      background: #2D8CEB;
      color: white;
      border: none;
      padding: 4px 10px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 11px;
      font-weight: 600;
    }
    .om-add-btn:hover {
      background: #4DA3F0;
    }
    .om-add-btn:disabled {
      background: #333;
      color: #666;
      cursor: default;
    }
    .om-add-slice-btn {
      background: #3a3a3a;
      color: #ccc;
    }
    .om-add-slice-btn:hover:not(:disabled) {
      background: #4a4a4a;
    }
    .om-target-items {
      flex: 1;
      overflow-y: auto;
      padding: 4px;
    }
    .om-empty {
      padding: 24px 16px;
      text-align: center;
      color: #666;
      font-size: 12px;
    }
    .om-target-item {
      padding: 8px 10px;
      margin-bottom: 2px;
      border-radius: 4px;
      cursor: pointer;
      border: 1px solid transparent;
    }
    .om-target-item:hover {
      background: #1e1e1e;
    }
    .om-target-item.selected {
      background: #1a2a3a;
      border-color: #2D8CEB;
    }
    .om-target-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .om-target-controls {
      margin-top: 6px;
    }
    .om-target-status {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .om-target-status.enabled {
      background: #4f4;
    }
    .om-target-status.disabled {
      background: #666;
    }
    .om-target-name {
      font-weight: 500;
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .om-target-type {
      font-size: 10px;
      color: #666;
      text-transform: uppercase;
    }
    .om-toggle-btn {
      background: #333;
      color: #888;
      border: 1px solid #444;
      padding: 2px 8px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 10px;
      font-weight: 600;
    }
    .om-toggle-btn.active {
      background: #1a3a1a;
      color: #4f4;
      border-color: #4f4;
    }
    .om-close-btn {
      background: none;
      color: #888;
      border: 1px solid #444;
      padding: 2px 6px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 10px;
    }
    .om-close-btn:hover {
      background: #4a1a1a;
      color: #f44;
      border-color: #f44;
    }

    /* Closed (deactivated) target state */
    .om-target-item.closed {
      opacity: 0.55;
    }
    .om-target-status.closed {
      background: #555;
      border: 1px solid #777;
    }
    .om-source-label-readonly {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      color: #777;
      font-style: italic;
    }
    .om-restore-btn {
      background: #2D8CEB;
      color: white;
      border: none;
      padding: 2px 10px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 10px;
      font-weight: 600;
    }
    .om-restore-btn:hover {
      background: #4DA3F0;
    }
    .om-remove-btn {
      background: none;
      color: #888;
      border: 1px solid #444;
      padding: 2px 6px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 10px;
    }
    .om-remove-btn:hover {
      background: #4a1a1a;
      color: #f44;
      border-color: #f44;
    }

    /* Source Selector */
    .om-source-selector {
      position: relative;
      flex: 1;
      min-width: 0;
    }
    .om-source-btn {
      display: flex;
      align-items: center;
      gap: 4px;
      width: 100%;
      background: #222;
      color: #ccc;
      border: 1px solid #444;
      padding: 3px 8px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 11px;
      text-align: left;
    }
    .om-source-btn:hover {
      border-color: #666;
    }
    .om-source-label {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .om-source-arrow {
      font-size: 8px;
      color: #888;
      flex-shrink: 0;
    }
    .om-source-dropdown {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      z-index: 100;
      background: #1e1e1e;
      border: 1px solid #444;
      border-radius: 4px;
      margin-top: 2px;
      max-height: 250px;
      overflow-y: auto;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .om-source-option {
      display: block;
      width: 100%;
      background: none;
      color: #ccc;
      border: none;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 11px;
      text-align: left;
    }
    .om-source-option:hover {
      background: #2a2a2a;
    }
    .om-source-option.active {
      background: #1a2a3a;
      color: #4DA3F0;
    }
    .om-source-separator {
      height: 1px;
      background: #333;
      margin: 2px 0;
    }

    /* Preview Wrapper (for overlay positioning) */
    .om-preview-wrapper {
      position: relative;
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 0;
    }

    /* Preview */
    .om-preview {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      box-sizing: border-box;
    }
    .om-preview-canvas {
      max-width: 100%;
      max-height: 100%;
      background: #000;
      object-fit: contain;
    }
    .om-preview-empty {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      height: 100%;
      color: #555;
      font-size: 14px;
    }

    /* Tab Bar */
    .om-tab-bar {
      display: flex;
      gap: 0;
      background: #161616;
      border-bottom: 1px solid #2a2a2a;
      flex-shrink: 0;
      padding: 0 12px;
    }
    .om-tab {
      background: none;
      border: none;
      color: #888;
      padding: 8px 16px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 2px solid transparent;
      transition: color 0.15s, border-color 0.15s;
    }
    .om-tab:hover {
      color: #ccc;
    }
    .om-tab.active {
      color: #2D8CEB;
      border-bottom-color: #2D8CEB;
    }

    /* Nested Slice Items (under each target) */
    .om-slice-items-nested {
      padding-left: 16px;
      border-left: 2px solid #2a2a2a;
      margin-left: 14px;
      margin-bottom: 4px;
    }
    .om-slice-item {
      padding: 6px 10px;
      margin-bottom: 1px;
      border-radius: 4px;
      cursor: pointer;
      border: 1px solid transparent;
      font-size: 12px;
    }
    .om-slice-item:hover {
      background: #1e1e1e;
    }
    .om-slice-item.selected {
      background: #1a2a3a;
      border-color: #2D8CEB;
    }
    .om-slice-item.disabled {
      opacity: 0.5;
    }
    .om-slice-row {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .om-target-status.small {
      width: 6px;
      height: 6px;
    }
    .om-slice-name {
      font-weight: 500;
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
    }
    .om-slice-mode {
      font-size: 10px;
      color: #666;
    }
    .om-slice-controls {
      display: flex;
      gap: 4px;
      margin-top: 4px;
    }

    /* Slice SVG Overlay */
    .om-slice-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .om-slice-overlay rect,
    .om-slice-overlay path,
    .om-slice-overlay circle {
      pointer-events: auto;
    }
  `;
}
