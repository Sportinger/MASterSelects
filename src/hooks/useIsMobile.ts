// Hook to detect mobile devices
// Only triggers on actual mobile devices, NOT on small desktop windows

import { useState, useEffect } from 'react';

/** Check if the device is actually mobile (phone/tablet), not just a small window */
function checkIsMobileDevice(): boolean {
  if (typeof window === 'undefined') return false;

  // Modern API - most reliable
  const nav = navigator as Navigator & { userAgentData?: { mobile: boolean } };
  if (nav.userAgentData) {
    return nav.userAgentData.mobile;
  }

  // Fallback: user agent string check
  return /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

export function useIsMobile(): boolean {
  const [isMobile] = useState(() => checkIsMobileDevice());

  // Device type doesn't change during session, no need for resize/orientation listeners
  return isMobile;
}

// Force mobile mode via URL param for testing: ?mobile=true
export function useForceMobile(): boolean {
  const [forceMobile, setForceMobile] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setForceMobile(params.get('mobile') === 'true');
  }, []);

  return forceMobile;
}
