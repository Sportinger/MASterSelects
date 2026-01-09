// Maps panel type to actual component

import type { PanelType } from '../../types/dock';
import { Preview } from '../preview';
import { EffectsPanel, ClipPropertiesPanel, LayerPanel, MediaPanel, MultiCamPanel, TranscriptPanel, AnalysisPanel, AIChatPanel } from '../panels';
import { ExportPanel } from '../export';
import { Timeline } from '../timeline';

interface DockPanelContentProps {
  type: PanelType;
}

export function DockPanelContent({ type }: DockPanelContentProps) {
  switch (type) {
    case 'preview':
      return <Preview />;
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
    case 'ai-chat':
      return <AIChatPanel />;
    default:
      return <div className="panel-placeholder">Unknown panel: {type}</div>;
  }
}
