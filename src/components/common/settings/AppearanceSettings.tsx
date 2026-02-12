import { useSettingsStore, type ThemeMode } from '../../../stores/settingsStore';

const themeOptions: { id: ThemeMode; label: string; bg: string; bar: string; accent: string }[] = [
  { id: 'dark',     label: 'Dark',     bg: '#1e1e1e', bar: '#0f0f0f', accent: '#2D8CEB' },
  { id: 'light',    label: 'Light',    bg: '#f5f5f5', bar: '#dedede', accent: '#1a73e8' },
  { id: 'midnight', label: 'Midnight', bg: '#000000', bar: '#111111', accent: '#3d9df5' },
  { id: 'system',   label: 'System',   bg: 'linear-gradient(135deg, #1e1e1e 50%, #f5f5f5 50%)', bar: '#333', accent: '#2D8CEB' },
  { id: 'crazy',    label: 'Crazy You', bg: 'linear-gradient(135deg, #e91e63 0%, #9c27b0 33%, #2196f3 66%, #4caf50 100%)', bar: 'linear-gradient(90deg, #ff9800, #e91e63)', accent: '#ffeb3b' },
];

export function AppearanceSettings() {
  const theme = useSettingsStore((s) => s.theme);
  const setTheme = useSettingsStore((s) => s.setTheme);

  return (
    <div className="settings-category-content">
      <h2>Appearance</h2>

      <div className="settings-group">
        <div className="settings-group-title">Theme</div>
        <div className="theme-selector">
          {themeOptions.map((opt) => (
            <label key={opt.id} className={`theme-card ${theme === opt.id ? 'active' : ''}`}>
              <input
                type="radio"
                name="theme"
                value={opt.id}
                checked={theme === opt.id}
                onChange={() => setTheme(opt.id)}
              />
              <div
                className="theme-preview"
                style={{ background: opt.bg }}
              >
                <div className="theme-preview-bar" style={{ background: opt.bar }} />
                <div className="theme-preview-accent" style={{ background: opt.accent }} />
              </div>
              <span className="theme-card-label">{opt.label}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
