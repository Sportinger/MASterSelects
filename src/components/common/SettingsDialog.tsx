// Settings Dialog - After Effects style preferences with sidebar navigation

import { useState, useCallback, useRef } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useDraggableDialog } from './settings/useDraggableDialog';
import { AppearanceSettings } from './settings/AppearanceSettings';
import { GeneralSettings } from './settings/GeneralSettings';
import { PreviewsSettings } from './settings/PreviewsSettings';
import { ImportSettings } from './settings/ImportSettings';
import { TranscriptionSettings } from './settings/TranscriptionSettings';
import { OutputSettings } from './settings/OutputSettings';
import { PerformanceSettings } from './settings/PerformanceSettings';
import { ApiKeysSettings } from './settings/ApiKeysSettings';
import './settings/SettingsDialog.css';

interface SettingsDialogProps {
  onClose: () => void;
}

type SettingsCategory =
  | 'appearance'
  | 'general'
  | 'previews'
  | 'import'
  | 'transcription'
  | 'output'
  | 'performance'
  | 'apiKeys';

interface CategoryConfig {
  id: SettingsCategory;
  label: string;
  icon: string;
}

const categories: CategoryConfig[] = [
  { id: 'appearance', label: 'Appearance', icon: '\uD83C\uDFA8' },
  { id: 'general', label: 'General', icon: '\u2699' },
  { id: 'previews', label: 'Previews', icon: '\u25B6' },
  { id: 'import', label: 'Import', icon: '\uD83D\uDCE5' },
  { id: 'transcription', label: 'Transcription', icon: '\uD83C\uDFA4' },
  { id: 'output', label: 'Output', icon: '\uD83D\uDCE4' },
  { id: 'performance', label: 'Performance', icon: '\u26A1' },
  { id: 'apiKeys', label: 'API Keys', icon: '\uD83D\uDD11' },
];

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('appearance');
  const dialogRef = useRef<HTMLDivElement>(null);
  const { position, isDragging, handleMouseDown } = useDraggableDialog(dialogRef);

  const { apiKeys, setApiKey } = useSettingsStore();

  // Local state for API keys (to avoid saving on every keystroke)
  const [localKeys, setLocalKeys] = useState<{ [key: string]: string }>({ ...apiKeys });

  const handleSave = useCallback(() => {
    Object.entries(localKeys).forEach(([provider, key]) => {
      setApiKey(provider as keyof typeof apiKeys, key);
    });
    onClose();
  }, [localKeys, setApiKey, onClose]);

  const handleKeyChange = (provider: string, value: string) => {
    setLocalKeys((prev) => ({ ...prev, [provider]: value }));
  };

  const renderCategoryContent = () => {
    switch (activeCategory) {
      case 'appearance': return <AppearanceSettings />;
      case 'general': return <GeneralSettings />;
      case 'previews': return <PreviewsSettings />;
      case 'import': return <ImportSettings />;
      case 'transcription': return <TranscriptionSettings localKeys={localKeys} />;
      case 'output': return <OutputSettings />;
      case 'performance': return <PerformanceSettings />;
      case 'apiKeys': return <ApiKeysSettings localKeys={localKeys} onKeyChange={handleKeyChange} />;
      default: return null;
    }
  };

  return (
    <div className="settings-container">
      <div
        ref={dialogRef}
        className={`settings-dialog ${isDragging ? 'dragging' : ''}`}
        style={{
          left: position.x,
          top: position.y,
        }}
      >
        {/* Header - Draggable */}
        <div
          className="settings-header"
          onMouseDown={handleMouseDown}
        >
          <h1>Preferences</h1>
          <button className="settings-close" onClick={onClose} onMouseDown={(e) => e.stopPropagation()}>{'\u00D7'}</button>
        </div>

        {/* Main content with sidebar */}
        <div className="settings-main">
          {/* Sidebar */}
          <div className="settings-sidebar">
            {categories.map((cat) => (
              <button
                key={cat.id}
                className={`sidebar-item ${activeCategory === cat.id ? 'active' : ''}`}
                onClick={() => setActiveCategory(cat.id)}
              >
                <span className="sidebar-icon">{cat.icon}</span>
                <span className="sidebar-label">{cat.label}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="settings-content">
            {renderCategoryContent()}
          </div>
        </div>

        {/* Footer */}
        <div className="settings-footer">
          <button className="btn-cancel" onClick={onClose}>Cancel</button>
          <button className="btn-save" onClick={handleSave}>OK</button>
        </div>
      </div>
    </div>
  );
}
