//! Periodic auto-save logic — tracks dirty state and save timing.
//!
//! `AutoSaver` does NOT own a thread or async task. Instead, it provides
//! a stateful timer that the caller polls (e.g., from the main loop or a
//! dedicated tick). When `should_save()` returns `true`, the caller should
//! trigger a save and call `mark_saved()`.

use std::time::{Duration, Instant};

use tracing::{debug, info};

/// Default auto-save interval in seconds.
pub const DEFAULT_AUTOSAVE_INTERVAL_SECS: u32 = 60;

/// Manages periodic auto-save timing and dirty-state tracking.
#[derive(Debug)]
pub struct AutoSaver {
    /// How often to auto-save (in seconds).
    interval: Duration,
    /// Whether the project has unsaved changes.
    dirty: bool,
    /// When the last successful save occurred.
    last_saved: Instant,
    /// Whether auto-save is enabled.
    enabled: bool,
}

impl AutoSaver {
    /// Create a new auto-saver with the given interval in seconds.
    ///
    /// The saver starts in a clean (not dirty) state.
    pub fn new(interval_secs: u32) -> Self {
        let interval = Duration::from_secs(interval_secs as u64);
        info!(interval_secs, "AutoSaver initialized");
        Self {
            interval,
            dirty: false,
            last_saved: Instant::now(),
            enabled: true,
        }
    }

    /// Check whether an auto-save should be triggered now.
    ///
    /// Returns `true` if all of the following are true:
    /// - Auto-save is enabled
    /// - The project has unsaved changes (dirty)
    /// - Enough time has elapsed since the last save
    pub fn should_save(&self) -> bool {
        if !self.enabled {
            return false;
        }
        if !self.dirty {
            return false;
        }
        self.last_saved.elapsed() >= self.interval
    }

    /// Mark the project as having unsaved changes.
    pub fn mark_dirty(&mut self) {
        if !self.dirty {
            debug!("Project marked as dirty");
        }
        self.dirty = true;
    }

    /// Mark the project as saved (clear dirty flag and reset timer).
    pub fn mark_saved(&mut self) {
        self.dirty = false;
        self.last_saved = Instant::now();
        debug!("Project marked as saved, timer reset");
    }

    /// Check if the project has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the configured auto-save interval.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Change the auto-save interval.
    pub fn set_interval(&mut self, secs: u32) {
        self.interval = Duration::from_secs(secs as u64);
        debug!(interval_secs = secs, "AutoSave interval updated");
    }

    /// Enable or disable auto-saving.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        debug!(enabled, "AutoSave enabled state changed");
    }

    /// Check if auto-save is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Time remaining until the next auto-save would be due.
    ///
    /// Returns `Duration::ZERO` if a save is already due.
    pub fn time_until_next_save(&self) -> Duration {
        let elapsed = self.last_saved.elapsed();
        self.interval.saturating_sub(elapsed)
    }
}

impl Default for AutoSaver {
    fn default() -> Self {
        Self::new(DEFAULT_AUTOSAVE_INTERVAL_SECS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_autosaver_not_dirty() {
        let saver = AutoSaver::new(60);
        assert!(!saver.is_dirty());
        assert!(!saver.should_save());
    }

    #[test]
    fn mark_dirty_then_check() {
        let mut saver = AutoSaver::new(60);
        saver.mark_dirty();
        assert!(saver.is_dirty());
        // Should not save yet because interval hasn't elapsed
        // (unless test takes 60+ seconds, which it won't)
        assert!(!saver.should_save());
    }

    #[test]
    fn mark_saved_clears_dirty() {
        let mut saver = AutoSaver::new(60);
        saver.mark_dirty();
        assert!(saver.is_dirty());
        saver.mark_saved();
        assert!(!saver.is_dirty());
        assert!(!saver.should_save());
    }

    #[test]
    fn should_save_after_interval() {
        // Use a zero-second interval so it triggers immediately
        let mut saver = AutoSaver::new(0);
        saver.mark_dirty();
        // With interval of 0, should_save should be true immediately
        assert!(saver.should_save());
    }

    #[test]
    fn should_not_save_when_disabled() {
        let mut saver = AutoSaver::new(0);
        saver.mark_dirty();
        saver.set_enabled(false);
        assert!(!saver.should_save());
        assert!(!saver.is_enabled());
    }

    #[test]
    fn should_not_save_when_clean() {
        let saver = AutoSaver::new(0);
        assert!(!saver.should_save());
    }

    #[test]
    fn set_interval_changes_duration() {
        let mut saver = AutoSaver::new(60);
        assert_eq!(saver.interval(), Duration::from_secs(60));
        saver.set_interval(120);
        assert_eq!(saver.interval(), Duration::from_secs(120));
    }

    #[test]
    fn default_interval() {
        let saver = AutoSaver::default();
        assert_eq!(
            saver.interval(),
            Duration::from_secs(DEFAULT_AUTOSAVE_INTERVAL_SECS as u64)
        );
    }

    #[test]
    fn time_until_next_save_decreases() {
        let saver = AutoSaver::new(3600); // 1 hour
        let remaining = saver.time_until_next_save();
        // Should be close to 3600 seconds (minus tiny elapsed time)
        assert!(remaining.as_secs() >= 3598);
    }

    #[test]
    fn time_until_next_save_zero_when_due() {
        let mut saver = AutoSaver::new(0);
        saver.mark_dirty();
        let remaining = saver.time_until_next_save();
        assert_eq!(remaining, Duration::ZERO);
    }

    #[test]
    fn mark_saved_resets_timer() {
        let mut saver = AutoSaver::new(0);
        saver.mark_dirty();
        assert!(saver.should_save());

        saver.mark_saved();
        assert!(!saver.should_save()); // Clean, even though interval is 0
    }

    #[test]
    fn enable_disable_toggle() {
        let mut saver = AutoSaver::new(60);
        assert!(saver.is_enabled());

        saver.set_enabled(false);
        assert!(!saver.is_enabled());

        saver.set_enabled(true);
        assert!(saver.is_enabled());
    }
}
