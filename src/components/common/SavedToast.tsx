// SavedToast - Center screen notification for save actions
// Shows a brief "Saved" message in yellow when project is saved

import { useEffect, useState } from 'react';

interface SavedToastProps {
  visible: boolean;
  onHide: () => void;
}

export function SavedToast({ visible, onHide }: SavedToastProps) {
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    if (visible) {
      setIsExiting(false);
      // Start exit animation after 1.5 seconds
      const exitTimer = setTimeout(() => {
        setIsExiting(true);
      }, 1500);

      // Hide completely after animation
      const hideTimer = setTimeout(() => {
        onHide();
      }, 1800);

      return () => {
        clearTimeout(exitTimer);
        clearTimeout(hideTimer);
      };
    }
  }, [visible, onHide]);

  if (!visible) return null;

  return (
    <div className={`saved-toast ${isExiting ? 'exiting' : ''}`}>
      Saved
    </div>
  );
}
