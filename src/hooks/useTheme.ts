import { useEffect, useMemo, useRef } from 'react';
import { useSettingsStore } from '../stores/settingsStore';
import type { ThemeMode } from '../stores/settingsStore';

type ResolvedTheme = 'dark' | 'light' | 'midnight' | 'crazy';

function resolveTheme(theme: ThemeMode): ResolvedTheme {
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return theme;
}

/** Generate a random hue-shifted color palette and apply as CSS custom properties */
function applyCrazyColors(root: HTMLElement) {
  const hue = () => Math.floor(Math.random() * 360);
  const sat = () => 40 + Math.floor(Math.random() * 40); // 40-80%
  const light = (base: number, range: number) => base + Math.floor(Math.random() * range);

  // Generate a random palette
  const h1 = hue(), h2 = hue(), h3 = hue();

  root.style.setProperty('--bg-primary', `hsl(${h1}, ${sat()}%, ${light(8, 10)}%)`);
  root.style.setProperty('--bg-secondary', `hsl(${h1}, ${sat()}%, ${light(12, 8)}%)`);
  root.style.setProperty('--bg-tertiary', `hsl(${h1}, ${sat()}%, ${light(16, 8)}%)`);
  root.style.setProperty('--bg-hover', `hsl(${h1}, ${sat()}%, ${light(22, 8)}%)`);
  root.style.setProperty('--bg-active', `hsl(${h1}, ${sat()}%, ${light(26, 6)}%)`);
  root.style.setProperty('--bg-elevated', `hsl(${h1}, ${sat()}%, ${light(18, 6)}%)`);
  root.style.setProperty('--bg-input', `hsl(${h1}, ${sat()}%, ${light(10, 6)}%)`);

  root.style.setProperty('--border-color', `hsl(${h1}, ${sat()}%, ${light(20, 10)}%)`);
  root.style.setProperty('--border-subtle', `hsl(${h1}, ${sat()}%, ${light(28, 10)}%)`);
  root.style.setProperty('--border-strong', `hsl(${h1}, ${sat()}%, ${light(35, 10)}%)`);

  root.style.setProperty('--text-primary', `hsl(${h2}, ${sat()}%, ${light(80, 15)}%)`);
  root.style.setProperty('--text-secondary', `hsl(${h2}, ${sat()}%, ${light(60, 15)}%)`);
  root.style.setProperty('--text-muted', `hsl(${h2}, ${sat()}%, ${light(45, 10)}%)`);

  root.style.setProperty('--accent', `hsl(${h3}, ${sat() + 15}%, ${light(50, 15)}%)`);
  root.style.setProperty('--accent-hover', `hsl(${h3}, ${sat() + 15}%, ${light(60, 10)}%)`);
  root.style.setProperty('--accent-dim', `hsla(${h3}, ${sat() + 15}%, 50%, 0.15)`);
  root.style.setProperty('--accent-subtle', `hsla(${h3}, ${sat() + 15}%, 50%, 0.1)`);
  root.style.setProperty('--accent-timeline', `hsl(${h3}, ${sat() + 15}%, ${light(50, 15)}%)`);

  root.style.setProperty('--tab-active-bg', `hsl(${h1}, ${sat()}%, ${light(24, 8)}%)`);
  root.style.setProperty('--scrollbar-thumb', `hsl(${h1}, ${sat()}%, ${light(30, 10)}%)`);
  root.style.setProperty('--scrollbar-thumb-hover', `hsl(${h1}, ${sat()}%, ${light(40, 10)}%)`);

  root.style.setProperty('--timeline-grid-video', `hsl(${h3}, ${sat()}%, ${light(14, 6)}%)`);
  root.style.setProperty('--timeline-grid-audio', `hsl(${(h3 + 120) % 360}, ${sat()}%, ${light(14, 6)}%)`);

  root.style.setProperty('--chat-user-bg', `hsl(${h2}, ${sat()}%, ${light(20, 10)}%)`);
  root.style.setProperty('--chat-user-border', `hsl(${h2}, ${sat()}%, ${light(28, 10)}%)`);

  const h4 = hue();
  root.style.setProperty('--danger', `hsl(${h4}, 70%, 50%)`);
  root.style.setProperty('--success', `hsl(${(h4 + 120) % 360}, 60%, 45%)`);
  root.style.setProperty('--warning', `hsl(${(h4 + 60) % 360}, 70%, 50%)`);
  root.style.setProperty('--purple', `hsl(${(h4 + 180) % 360}, 60%, 55%)`);

  root.style.setProperty('--shadow-md', `0 4px 16px hsla(${h1}, 80%, 20%, 0.5)`);
  root.style.setProperty('--shadow-lg', `0 8px 32px hsla(${h1}, 80%, 20%, 0.5)`);
}

/** Remove all inline style overrides set by crazy theme */
function clearCrazyColors(root: HTMLElement) {
  const props = [
    '--bg-primary', '--bg-secondary', '--bg-tertiary', '--bg-hover', '--bg-active',
    '--bg-elevated', '--bg-input', '--border-color', '--border-subtle', '--border-strong',
    '--text-primary', '--text-secondary', '--text-muted', '--accent', '--accent-hover',
    '--accent-dim', '--accent-subtle', '--accent-timeline', '--tab-active-bg',
    '--scrollbar-thumb', '--scrollbar-thumb-hover', '--timeline-grid-video',
    '--timeline-grid-audio', '--chat-user-bg', '--chat-user-border',
    '--danger', '--success', '--warning', '--purple', '--shadow-md', '--shadow-lg',
  ];
  for (const prop of props) {
    root.style.removeProperty(prop);
  }
}

export function useTheme() {
  const theme = useSettingsStore((s) => s.theme);
  const setTheme = useSettingsStore((s) => s.setTheme);
  const crazyInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  const resolvedTheme = useMemo(() => resolveTheme(theme), [theme]);

  useEffect(() => {
    const root = document.documentElement;
    const resolved = resolveTheme(theme);

    // Clean up previous crazy interval
    if (crazyInterval.current) {
      clearInterval(crazyInterval.current);
      crazyInterval.current = null;
    }

    // Clear any inline crazy overrides before switching
    clearCrazyColors(root);

    // Add transition class for smooth switching
    root.classList.add('theme-transitioning');
    root.dataset.theme = resolved;

    if (theme === 'crazy') {
      // Apply random colors immediately, then re-randomize every 5 seconds
      applyCrazyColors(root);
      crazyInterval.current = setInterval(() => {
        root.classList.add('theme-transitioning');
        applyCrazyColors(root);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => root.classList.remove('theme-transitioning'));
        });
      }, 5000);
    }

    requestAnimationFrame(() => {
      requestAnimationFrame(() => root.classList.remove('theme-transitioning'));
    });

    // Listen for OS changes when in system mode
    if (theme === 'system') {
      const mql = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = () => {
        root.dataset.theme = resolveTheme(theme);
      };
      mql.addEventListener('change', handler);
      return () => mql.removeEventListener('change', handler);
    }

    return () => {
      if (crazyInterval.current) {
        clearInterval(crazyInterval.current);
        crazyInterval.current = null;
      }
    };
  }, [theme]);

  return { theme, resolvedTheme, setTheme };
}
