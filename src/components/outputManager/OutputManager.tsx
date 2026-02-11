// OutputManager - main component for managing output targets
// Renders in a popup window, manages source routing for all output windows

import { useState } from 'react';
import { TargetList } from './TargetList';
import { TargetPreview } from './TargetPreview';
import { TabBar } from './TabBar';
import { SliceInputOverlay } from './SliceInputOverlay';
import { SliceOutputOverlay } from './SliceOutputOverlay';
import { useSliceStore } from '../../stores/sliceStore';

export function OutputManager() {
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);
  const activeTab = useSliceStore((s) => s.activeTab);

  return (
    <div className="om-container">
      <div className="om-header">
        <h2 className="om-title">Output Manager</h2>
      </div>
      <div className="om-body">
        <div className="om-main">
          <TabBar />
          <div className="om-preview-wrapper">
            <TargetPreview targetId={selectedTargetId} />
            {selectedTargetId && activeTab === 'input' && (
              <SliceInputOverlay targetId={selectedTargetId} width={1920} height={1080} />
            )}
            {selectedTargetId && activeTab === 'output' && (
              <SliceOutputOverlay targetId={selectedTargetId} width={1920} height={1080} />
            )}
          </div>
        </div>
        <div className="om-sidebar">
          <TargetList
            selectedTargetId={selectedTargetId}
            onSelect={setSelectedTargetId}
          />
        </div>
      </div>
    </div>
  );
}
