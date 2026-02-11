// SliceList - sidebar panel for managing slices on the selected target

import { useSliceStore } from '../../stores/sliceStore';

interface SliceListProps {
  targetId: string | null;
}

export function SliceList({ targetId }: SliceListProps) {
  const configs = useSliceStore((s) => s.configs);
  const addSlice = useSliceStore((s) => s.addSlice);
  const removeSlice = useSliceStore((s) => s.removeSlice);
  const selectSlice = useSliceStore((s) => s.selectSlice);
  const setSliceEnabled = useSliceStore((s) => s.setSliceEnabled);
  const resetSliceWarp = useSliceStore((s) => s.resetSliceWarp);

  if (!targetId) {
    return (
      <div className="om-slice-list">
        <div className="om-slice-list-header">
          <span className="om-slice-list-title">Slices</span>
        </div>
        <div className="om-empty">Select a target first</div>
      </div>
    );
  }

  const config = configs.get(targetId);
  const slices = config?.slices ?? [];
  const selectedSliceId = config?.selectedSliceId ?? null;

  return (
    <div className="om-slice-list">
      <div className="om-slice-list-header">
        <span className="om-slice-list-title">Slices</span>
        <button
          className="om-add-btn"
          onClick={() => addSlice(targetId)}
        >
          + Add Slice
        </button>
      </div>
      <div className="om-slice-items">
        {slices.length === 0 && (
          <div className="om-empty">No slices. Add one to start warping.</div>
        )}
        {slices.map((slice) => (
          <div
            key={slice.id}
            className={`om-slice-item ${selectedSliceId === slice.id ? 'selected' : ''} ${!slice.enabled ? 'disabled' : ''}`}
            onClick={() => selectSlice(targetId, slice.id)}
          >
            <div className="om-slice-row">
              <span className={`om-target-status ${slice.enabled ? 'enabled' : 'disabled'}`} />
              <span className="om-slice-name">{slice.name}</span>
              <span className="om-slice-mode">Corner Pin</span>
            </div>
            <div className="om-slice-controls">
              <button
                className={`om-toggle-btn ${slice.enabled ? 'active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setSliceEnabled(targetId, slice.id, !slice.enabled);
                }}
              >
                {slice.enabled ? 'ON' : 'OFF'}
              </button>
              <button
                className="om-close-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  resetSliceWarp(targetId, slice.id);
                }}
                title="Reset warp"
              >
                Reset
              </button>
              <button
                className="om-remove-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  removeSlice(targetId, slice.id);
                }}
                title="Delete slice"
              >
                Del
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
