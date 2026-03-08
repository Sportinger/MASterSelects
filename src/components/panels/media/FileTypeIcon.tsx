// File-type icons (AE style) - inline SVGs
// Small (14px) for list view, large for grid thumbnails
import React, { memo } from 'react';

interface FileTypeIconProps {
  type?: string;
  /** Render larger version for grid thumbnail placeholder */
  large?: boolean;
}

export const FileTypeIcon = memo(({ type, large }: FileTypeIconProps) => {
  const size = large ? 48 : 14;
  const style: React.CSSProperties = { width: size, height: size, flexShrink: 0, display: 'block' };

  if (large) {
    return <LargeIcon type={type} style={style} />;
  }

  switch (type) {
    case 'video':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="3" width="14" height="10" rx="1.5" fill="#4a6fa5" stroke="#6b9bd2" strokeWidth="0.7"/>
          <rect x="3" y="5" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="7" y="5" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="11" y="5" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="3" y="9" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="7" y="9" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="11" y="9" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
        </svg>
      );
    case 'audio':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#4a7a4a" stroke="#6aaa6a" strokeWidth="0.7"/>
          <path d="M4 6v4M6 5v6M8 4v8M10 5v6M12 6v4" stroke="#8fdf8f" strokeWidth="1.2" strokeLinecap="round"/>
        </svg>
      );
    case 'image':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#5a6a8a" stroke="#7a9aba" strokeWidth="0.7"/>
          <circle cx="5.5" cy="6" r="1.5" fill="#aaccee"/>
          <path d="M1.5 11l3.5-3 2.5 2 3-4 4 5v0.5c0 .55-.45 1-1 1h-12c-.55 0-1-.45-1-1z" fill="#7a9aba" opacity="0.8"/>
        </svg>
      );
    case 'composition':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#7a5a8a" stroke="#aa7abb" strokeWidth="0.7"/>
          <circle cx="8" cy="8" r="3.5" stroke="#cc99dd" strokeWidth="1" fill="none"/>
          <circle cx="8" cy="8" r="1" fill="#cc99dd"/>
        </svg>
      );
    case 'text':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#8a6a5a" stroke="#bb9a7a" strokeWidth="0.7"/>
          <text x="8" y="11.5" textAnchor="middle" fill="#eeddcc" fontSize="9" fontWeight="bold" fontFamily="sans-serif">T</text>
        </svg>
      );
    case 'solid':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#777" stroke="#999" strokeWidth="0.7"/>
          <rect x="4" y="5" width="8" height="6" rx="0.5" fill="#bbb"/>
        </svg>
      );
    default:
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <path d="M4 1.5h5.5l4 4V14c0 .55-.45 1-1 1H4c-.55 0-1-.45-1-1V2.5c0-.55.45-1 1-1z" fill="#5a5a5a" stroke="#888" strokeWidth="0.7"/>
          <path d="M9.5 1.5v4h4" stroke="#888" strokeWidth="0.7" fill="#6a6a6a"/>
        </svg>
      );
  }
});

/** Large grid icons — more detailed, nicer looking */
const LargeIcon = memo(({ type, style }: { type?: string; style: React.CSSProperties }) => {
  switch (type) {
    case 'folder':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <path d="M6 12C6 10.3431 7.34315 9 9 9H18.5858C19.1162 9 19.6249 9.21071 20 9.58579L22.4142 12H39C40.6569 12 42 13.3431 42 15V36C42 37.6569 40.6569 39 39 39H9C7.34315 39 6 37.6569 6 36V12Z" fill="#5c5c6e"/>
          <path d="M6 16H42V36C42 37.6569 40.6569 39 39 39H9C7.34315 39 6 37.6569 6 36V16Z" fill="#7a7a8e"/>
          <path d="M6 16H42V18H6V16Z" fill="#8e8e9e" opacity="0.4"/>
        </svg>
      );
    case 'composition':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          {/* Film strip background */}
          <rect x="4" y="8" width="40" height="32" rx="3" fill="#5a3d6e"/>
          <rect x="4" y="8" width="40" height="32" rx="3" stroke="#8a5faa" strokeWidth="1"/>
          {/* Sprocket holes */}
          <rect x="7" y="11" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="7" y="17" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="7" y="23" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="7" y="29" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="7" y="35" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="38" y="11" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="38" y="17" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="38" y="23" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="38" y="29" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          <rect x="38" y="35" width="3" height="3" rx="0.5" fill="#3a2548" opacity="0.6"/>
          {/* Inner frame */}
          <rect x="13" y="12" width="22" height="24" rx="1" fill="#4a2d5e" stroke="#7a4f9a" strokeWidth="0.5"/>
          {/* Composition target crosshair */}
          <circle cx="24" cy="24" r="8" stroke="#cc99ee" strokeWidth="1.2" fill="none" opacity="0.8"/>
          <circle cx="24" cy="24" r="3" stroke="#cc99ee" strokeWidth="0.8" fill="none" opacity="0.6"/>
          <circle cx="24" cy="24" r="1.2" fill="#cc99ee"/>
          <line x1="24" y1="14" x2="24" y2="19" stroke="#cc99ee" strokeWidth="0.6" opacity="0.4"/>
          <line x1="24" y1="29" x2="24" y2="34" stroke="#cc99ee" strokeWidth="0.6" opacity="0.4"/>
          <line x1="15" y1="24" x2="20" y2="24" stroke="#cc99ee" strokeWidth="0.6" opacity="0.4"/>
          <line x1="28" y1="24" x2="33" y2="24" stroke="#cc99ee" strokeWidth="0.6" opacity="0.4"/>
        </svg>
      );
    case 'audio':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <rect x="4" y="8" width="40" height="32" rx="3" fill="#2d5a3d"/>
          <rect x="4" y="8" width="40" height="32" rx="3" stroke="#4a9a5a" strokeWidth="1"/>
          {/* Waveform — organic, asymmetric for realism */}
          <path d="M8 24v0M10 22v4M12 20v8M14 18v12M16 21v6M18 17v14M20 19v10M22 15v18M24 13v22M26 16v16M28 14v20M30 18v12M32 20v8M34 16v16M36 19v10M38 22v4M40 23v2"
            stroke="#6edf7e" strokeWidth="1.8" strokeLinecap="round" opacity="0.9"/>
          {/* Subtle reflection */}
          <path d="M8 24M10 24.5v-1M12 25v-2M14 26v-4M16 25v-2M18 27v-6M20 26v-4M22 28v-8M24 29v-10M26 27v-6M28 28v-8M30 26v-4M32 25v-2M34 27v-6M36 26v-4M38 25v-2M40 24.5v-1"
            stroke="#4aba5a" strokeWidth="1.8" strokeLinecap="round" opacity="0.25"/>
        </svg>
      );
    case 'video':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <rect x="4" y="8" width="40" height="32" rx="3" fill="#3a5a8a"/>
          <rect x="4" y="8" width="40" height="32" rx="3" stroke="#5a8aba" strokeWidth="1"/>
          {/* Film grid */}
          <rect x="8" y="12" width="10" height="8" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="20" y="12" width="10" height="8" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="32" y="12" width="10" height="8" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="8" y="22" width="10" height="8" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="20" y="22" width="10" height="8" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="32" y="22" width="10" height="8" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="8" y="32" width="10" height="6" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="20" y="32" width="10" height="6" rx="1" fill="#2a4570" opacity="0.6"/>
          <rect x="32" y="32" width="10" height="6" rx="1" fill="#2a4570" opacity="0.6"/>
        </svg>
      );
    case 'image':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <rect x="4" y="8" width="40" height="32" rx="3" fill="#4a5a7a"/>
          <rect x="4" y="8" width="40" height="32" rx="3" stroke="#6a8aaa" strokeWidth="1"/>
          <circle cx="16" cy="18" r="4" fill="#8abbee" opacity="0.7"/>
          <path d="M4.5 32l10-8 7 5.5 8.5-11L44 35v3c0 1.38-1.12 2.5-2.5 2.5h-35c-1.38 0-2.5-1.12-2.5-2.5V32z" fill="#6a8aaa" opacity="0.6"/>
        </svg>
      );
    case 'text':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <rect x="4" y="8" width="40" height="32" rx="3" fill="#7a5a4a"/>
          <rect x="4" y="8" width="40" height="32" rx="3" stroke="#aa8a6a" strokeWidth="1"/>
          <text x="24" y="32" textAnchor="middle" fill="#eeddcc" fontSize="22" fontWeight="bold" fontFamily="sans-serif">T</text>
        </svg>
      );
    case 'solid':
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <rect x="4" y="8" width="40" height="32" rx="3" fill="#666"/>
          <rect x="4" y="8" width="40" height="32" rx="3" stroke="#888" strokeWidth="1"/>
          <rect x="12" y="16" width="24" height="16" rx="1" fill="#aaa"/>
        </svg>
      );
    default:
      return (
        <svg style={style} viewBox="0 0 48 48" fill="none">
          <path d="M14 6h14l12 12V40c0 1.1-.9 2-2 2H14c-1.1 0-2-.9-2-2V8c0-1.1.9-2 2-2z" fill="#4a4a4a" stroke="#777" strokeWidth="1"/>
          <path d="M28 6v12h12" stroke="#777" strokeWidth="1" fill="#5a5a5a"/>
        </svg>
      );
  }
});
