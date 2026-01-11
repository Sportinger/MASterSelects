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
      // Start exit animation after brief display (400ms)
      const exitTimer = setTimeout(() => {
        setIsExiting(true);
      }, 400);

      // Hide completely after fade out animation (400ms + 200ms = 600ms)
      const hideTimer = setTimeout(() => {
        onHide();
      }, 600);

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
