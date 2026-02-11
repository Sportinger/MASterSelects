// TabBar - Input/Output tab switcher for the Output Manager preview

import { useSliceStore } from '../../stores/sliceStore';

export function TabBar() {
  const activeTab = useSliceStore((s) => s.activeTab);
  const setActiveTab = useSliceStore((s) => s.setActiveTab);

  return (
    <div className="om-tab-bar">
      <button
        className={`om-tab ${activeTab === 'input' ? 'active' : ''}`}
        onClick={() => setActiveTab('input')}
      >
        Input
      </button>
      <button
        className={`om-tab ${activeTab === 'output' ? 'active' : ''}`}
        onClick={() => setActiveTab('output')}
      >
        Output
      </button>
    </div>
  );
}
