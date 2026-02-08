import type { ScopeViewMode } from './useScopeAnalysis';
import './ScopesPanel.css';

interface ScopeModeToolbarProps {
  mode: ScopeViewMode;
  onModeChange: (mode: ScopeViewMode) => void;
}

const MODES: { id: ScopeViewMode; label: string; className: string }[] = [
  { id: 'rgb', label: 'RGB', className: 'scope-mode-rgb' },
  { id: 'r', label: 'R', className: 'scope-mode-r' },
  { id: 'g', label: 'G', className: 'scope-mode-g' },
  { id: 'b', label: 'B', className: 'scope-mode-b' },
  { id: 'luma', label: 'Y', className: 'scope-mode-luma' },
];

export function ScopeModeToolbar({ mode, onModeChange }: ScopeModeToolbarProps) {
  return (
    <div className="scope-mode-toolbar">
      {MODES.map((m) => (
        <button
          key={m.id}
          className={`scope-mode-btn ${m.className} ${mode === m.id ? 'active' : ''}`}
          onClick={() => onModeChange(m.id)}
          title={m.id === 'luma' ? 'Luma (BT.709)' : m.label}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
