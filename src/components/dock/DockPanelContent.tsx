// Maps panel type to actual component
// Note: Effects, Transcript, Analysis are now integrated into PropertiesPanel

import { lazy, Suspense } from 'react';
import type { DockPanel, PreviewPanelData } from '../../types/dock';
import { Preview } from '../preview';
import { PropertiesPanel, MediaPanel } from '../panels';
import { Timeline } from '../timeline';

// Lazy-loaded panels: only loaded when the user opens them
// This keeps the initial bundle small by deferring export pipeline,
// AI services, YouTube API, and multicam analysis code
const ExportPanel = lazy(() => import('../export/ExportPanel').then(m => ({ default: m.ExportPanel })));
const MultiCamPanel = lazy(() => import('../panels/MultiCamPanel').then(m => ({ default: m.MultiCamPanel })));
const AIChatPanel = lazy(() => import('../panels/AIChatPanel').then(m => ({ default: m.AIChatPanel })));
const AIVideoPanel = lazy(() => import('../panels/AIVideoPanel').then(m => ({ default: m.AIVideoPanel })));
const YouTubePanel = lazy(() => import('../panels/YouTubePanel').then(m => ({ default: m.YouTubePanel })));
const TransitionsPanel = lazy(() => import('../panels/TransitionsPanel').then(m => ({ default: m.TransitionsPanel })));
const WaveformPanel = lazy(() => import('../panels/scopes/WaveformPanel').then(m => ({ default: m.WaveformPanel })));
const HistogramPanel = lazy(() => import('../panels/scopes/HistogramPanel').then(m => ({ default: m.HistogramPanel })));
const VectorscopePanel = lazy(() => import('../panels/scopes/VectorscopePanel').then(m => ({ default: m.VectorscopePanel })));

function PanelLoading() {
  return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Loading...</div>;
}

interface DockPanelContentProps {
  panel: DockPanel;
}

export function DockPanelContent({ panel }: DockPanelContentProps) {
  switch (panel.type) {
    case 'preview': {
      const previewData = panel.data as PreviewPanelData | undefined;
      return <Preview panelId={panel.id} compositionId={previewData?.compositionId ?? null} />;
    }
    case 'export':
      return <Suspense fallback={<PanelLoading />}><ExportPanel /></Suspense>;
    case 'clip-properties':
      return <PropertiesPanel />;
    case 'timeline':
      return <Timeline />;
    case 'media':
      return <MediaPanel />;
    case 'multicam':
      return <Suspense fallback={<PanelLoading />}><MultiCamPanel /></Suspense>;
    case 'ai-chat':
      return <Suspense fallback={<PanelLoading />}><AIChatPanel /></Suspense>;
    case 'ai-video':
      return <Suspense fallback={<PanelLoading />}><AIVideoPanel /></Suspense>;
    case 'youtube':
      return <Suspense fallback={<PanelLoading />}><YouTubePanel /></Suspense>;
    case 'transitions':
      return <Suspense fallback={<PanelLoading />}><TransitionsPanel /></Suspense>;
    case 'scope-waveform':
      return <Suspense fallback={<PanelLoading />}><WaveformPanel /></Suspense>;
    case 'scope-histogram':
      return <Suspense fallback={<PanelLoading />}><HistogramPanel /></Suspense>;
    case 'scope-vectorscope':
      return <Suspense fallback={<PanelLoading />}><VectorscopePanel /></Suspense>;
    default:
      return <div className="panel-placeholder">Unknown panel: {panel.type}</div>;
  }
}
