// Maps panel type to actual component

import type { PanelType } from '../../types/dock';
import { Preview } from '../Preview';
import { EffectsPanel } from '../EffectsPanel';
import { ClipPropertiesPanel } from '../ClipPropertiesPanel';
import { LayerPanel } from '../LayerPanel';
import { Timeline } from '../Timeline';

interface DockPanelContentProps {
  type: PanelType;
}

export function DockPanelContent({ type }: DockPanelContentProps) {
  switch (type) {
    case 'preview':
      return <Preview />;
    case 'effects':
      return <EffectsPanel />;
    case 'clip-properties':
      return <ClipPropertiesPanel />;
    case 'slots':
      return <LayerPanel />;
    case 'timeline':
      return <Timeline />;
    default:
      return <div className="panel-placeholder">Unknown panel: {type}</div>;
  }
}
