// App version - INCREMENT ON EVERY COMMIT!
// Format: MAJOR.MINOR.PATCH
// Increment PATCH (0.0.X) for each commit
export const APP_VERSION = '1.2.4';

// Build/Platform notice shown at top of changelog (set to null to hide)
export const BUILD_NOTICE: {
  type: 'info' | 'warning' | 'success';
  title: string;
  message: string;
} | null = null;

// Change entry type (used by UI)
export interface ChangeEntry {
  type: 'new' | 'fix' | 'improve' | 'refactor';
  title: string;
  description?: string;
  section?: string; // Optional section header to create visual dividers
  commits?: string[]; // Git commit hashes for linking to GitHub
}

// Time-grouped changelog entry
export interface TimeGroupedChanges {
  label: string; // "Today", "Last Week", "This Month", "Earlier"
  dateRange: string; // "Jan 20" or "Jan 13-19" etc
  changes: ChangeEntry[];
}

// Raw changelog entry as stored in changelog-data.json
export interface RawChangeEntry {
  date: string; // ISO date string YYYY-MM-DD
  type: 'new' | 'fix' | 'improve' | 'refactor';
  title: string;
  description?: string;
  section?: string;
  commits?: string[];
}

// Import changelog data from JSON
import changelogData from './changelog-data.json';
const RAW_CHANGELOG: RawChangeEntry[] = changelogData as RawChangeEntry[];

// Calculate relative time labels based on current date
function getTimeLabel(date: Date): { label: string; sortOrder: number } {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  if (date >= today) {
    return { label: 'Today', sortOrder: 0 };
  } else if (date >= yesterday) {
    return { label: 'Yesterday', sortOrder: 1 };
  } else if (date >= weekAgo) {
    return { label: 'Last Week', sortOrder: 2 };
  } else if (date >= monthAgo) {
    return { label: 'This Month', sortOrder: 3 };
  } else {
    return { label: 'Earlier', sortOrder: 4 };
  }
}

// Group changes by time period
export function getGroupedChangelog(): TimeGroupedChanges[] {
  const groups = new Map<string, { sortOrder: number; dateRange: string; changes: ChangeEntry[] }>();

  for (const entry of RAW_CHANGELOG) {
    const date = new Date(entry.date);
    const { label, sortOrder } = getTimeLabel(date);

    if (!groups.has(label)) {
      // Format date range
      const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      groups.set(label, { sortOrder, dateRange: dateStr, changes: [] });
    }

    const group = groups.get(label)!;
    group.changes.push({
      type: entry.type,
      title: entry.title,
      description: entry.description,
      section: entry.section,
      commits: entry.commits,
    });

    // Update date range if needed
    const currentDate = new Date(entry.date);
    const dateStr = currentDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    if (!group.dateRange.includes('-') && dateStr !== group.dateRange) {
      // Create a range
      const [firstPart] = group.dateRange.split(' ');
      const [, secondDay] = dateStr.split(' ');
      if (firstPart === dateStr.split(' ')[0]) {
        // Same month
        group.dateRange = `${group.dateRange.split(' ')[0]} ${secondDay}-${group.dateRange.split(' ')[1]}`;
      }
    }
  }

  // Sort groups by sortOrder and return
  return Array.from(groups.entries())
    .map(([label, data]) => ({
      label,
      dateRange: data.dateRange,
      changes: data.changes,
    }))
    .sort((a, b) => {
      const orderA = groups.get(a.label)?.sortOrder ?? 99;
      const orderB = groups.get(b.label)?.sortOrder ?? 99;
      return orderA - orderB;
    });
}

// Known issues and bugs - shown in What's New dialog
// Remove items when fixed
export const KNOWN_ISSUES: string[] = [
  'YouTube download requires Native Helper with yt-dlp installed',
  'Audio waveforms may not display for some video formats',
  'Very long videos (>2 hours) may cause performance issues',
];
