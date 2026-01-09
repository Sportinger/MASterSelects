// Maps panel type to actual component

import type { DockPanel, PreviewPanelData } from '../../types/dock';
import { Preview } from '../preview';
import { EffectsPanel, ClipPropertiesPanel, LayerPanel, MediaPanel, MultiCamPanel, TranscriptPanel, AnalysisPanel, AudioPanel, AIChatPanel } from '../panels';
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
    case 'effects':
      return <EffectsPanel />;
    case 'export':
      return <ExportPanel />;
    case 'clip-properties':
      return <ClipPropertiesPanel />;
    case 'slots':
      return <LayerPanel />;
    case 'timeline':
      return <Timeline />;
    case 'media':
      return <MediaPanel />;
    case 'multicam':
      return <MultiCamPanel />;
    case 'transcript':
      return <TranscriptPanel />;
    case 'analysis':
      return <AnalysisPanel />;
    case 'audio':
      return <AudioPanel />;
    case 'ai-chat':
      return <AIChatPanel />;
    default:
      return <div className="panel-placeholder">Unknown panel: {panel.type}</div>;
  }
}
