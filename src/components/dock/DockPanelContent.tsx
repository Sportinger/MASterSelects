// Maps panel type to actual component
// Note: Effects, Transcript, Analysis are now integrated into PropertiesPanel

import type { DockPanel, PreviewPanelData } from '../../types/dock';
import { Preview } from '../preview';
import { PropertiesPanel, LayerPanel, MediaPanel, MultiCamPanel, AIChatPanel, AIVideoPanel, YouTubePanel } from '../panels';
import { ExportPanel } from '../export';
import { Timeline } from '../timeline';

interface DockPanelContentProps {
  panel: DockPanel;
}

export function DockPanelContent({ panel }: DockPanelContentProps) {
  switch (panel.type) {
    case 'preview':
      const previewData = panel.data as PreviewPanelData | undefined;
      return <Preview panelId={panel.id} compositionId={previewData?.compositionId ?? null} />;
    case 'export':
      return <ExportPanel />;
    case 'clip-properties':
      return <PropertiesPanel />;
    case 'slots':
      return <LayerPanel />;
    case 'timeline':
      return <Timeline />;
    case 'media':
      return <MediaPanel />;
    case 'multicam':
      return <MultiCamPanel />;
    case 'ai-chat':
      return <AIChatPanel />;
    case 'ai-video':
      return <AIVideoPanel />;
    case 'youtube':
      return <YouTubePanel />;
    default:
      return <div className="panel-placeholder">Unknown panel: {panel.type}</div>;
  }
}
