//! Playback state management: play/pause, scrubbing, in/out points, loop, rate.

use ms_common::TimeCode;
use serde::{Deserialize, Serialize};

/// Current playback mode.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackMode {
    /// Playback is fully stopped (playhead at beginning or last stop position).
    #[default]
    Stopped,
    /// Actively playing forward at the configured rate.
    Playing,
    /// Paused at current position (can resume).
    Paused,
    /// User is actively scrubbing the playhead.
    Scrubbing,
}

/// Playback state for the timeline, including transport controls,
/// in/out points, loop, and variable speed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlaybackState {
    /// Current playback mode (Stopped, Playing, Paused, Scrubbing).
    pub mode: PlaybackMode,
    /// Current playhead position in seconds.
    pub current_time: TimeCode,
    /// Optional in-point for playback range.
    pub in_point: Option<TimeCode>,
    /// Optional out-point for playback range.
    pub out_point: Option<TimeCode>,
    /// Whether playback loops between in/out points (or start/end).
    pub loop_enabled: bool,
    /// Playback speed multiplier: 1.0 = normal, 0.5 = half, 2.0 = double.
    pub playback_rate: f64,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaybackState {
    /// Create a new playback state in stopped mode at time 0.
    pub fn new() -> Self {
        Self {
            mode: PlaybackMode::Stopped,
            current_time: TimeCode::ZERO,
            in_point: None,
            out_point: None,
            loop_enabled: false,
            playback_rate: 1.0,
        }
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        self.mode = PlaybackMode::Playing;
        tracing::debug!(time = %self.current_time, rate = self.playback_rate, "Playback started");
    }

    /// Pause playback at current position.
    pub fn pause(&mut self) {
        self.mode = PlaybackMode::Paused;
        tracing::debug!(time = %self.current_time, "Playback paused");
    }

    /// Stop playback. Resets playhead to the in-point (if set) or time 0.
    pub fn stop(&mut self) {
        self.mode = PlaybackMode::Stopped;
        self.current_time = self.in_point.unwrap_or(TimeCode::ZERO);
        tracing::debug!(time = %self.current_time, "Playback stopped");
    }

    /// Toggle between playing and paused. If stopped, starts playing.
    pub fn toggle_play_pause(&mut self) {
        match self.mode {
            PlaybackMode::Playing => self.pause(),
            PlaybackMode::Paused | PlaybackMode::Stopped => self.play(),
            PlaybackMode::Scrubbing => {
                // End scrub, then play
                self.play();
            }
        }
    }

    /// Seek to a specific time. Sets mode to paused if currently stopped.
    pub fn seek(&mut self, time: TimeCode) {
        self.current_time = time;
        if self.mode == PlaybackMode::Stopped {
            self.mode = PlaybackMode::Paused;
        }
        tracing::debug!(time = %self.current_time, "Seeked");
    }

    /// Begin scrubbing mode at the given time.
    pub fn start_scrub(&mut self, time: TimeCode) {
        self.mode = PlaybackMode::Scrubbing;
        self.current_time = time;
    }

    /// Update position during scrubbing.
    pub fn scrub_to(&mut self, time: TimeCode) {
        if self.mode == PlaybackMode::Scrubbing {
            self.current_time = time;
        }
    }

    /// End scrubbing, transition to paused at the current position.
    pub fn end_scrub(&mut self) {
        if self.mode == PlaybackMode::Scrubbing {
            self.mode = PlaybackMode::Paused;
            tracing::debug!(time = %self.current_time, "Scrub ended");
        }
    }

    /// Set the in-point for playback range.
    pub fn set_in_point(&mut self, time: TimeCode) {
        self.in_point = Some(time);
        tracing::debug!(in_point = %time, "In-point set");
    }

    /// Set the out-point for playback range.
    pub fn set_out_point(&mut self, time: TimeCode) {
        self.out_point = Some(time);
        tracing::debug!(out_point = %time, "Out-point set");
    }

    /// Clear both in-point and out-point.
    pub fn clear_in_out(&mut self) {
        self.in_point = None;
        self.out_point = None;
        tracing::debug!("In/out points cleared");
    }

    /// Enable or disable loop playback.
    pub fn set_loop(&mut self, enabled: bool) {
        self.loop_enabled = enabled;
        tracing::debug!(loop_enabled = enabled, "Loop toggled");
    }

    /// Set the playback rate. Clamped to [0.1, 16.0].
    pub fn set_rate(&mut self, rate: f64) {
        self.playback_rate = rate.clamp(0.1, 16.0);
        tracing::debug!(rate = self.playback_rate, "Playback rate set");
    }

    /// Get the effective playback range considering in/out points.
    /// Returns (start, end) where start is the in-point (or 0)
    /// and end is the out-point (or the given duration).
    pub fn playback_range(&self, duration: TimeCode) -> (TimeCode, TimeCode) {
        let start = self.in_point.unwrap_or(TimeCode::ZERO);
        let end = self.out_point.unwrap_or(duration);
        (start, end)
    }

    /// Whether playback is currently active (Playing or Scrubbing).
    pub fn is_active(&self) -> bool {
        matches!(self.mode, PlaybackMode::Playing | PlaybackMode::Scrubbing)
    }

    /// Whether the playhead is at or past the end of the playback range.
    pub fn is_at_end(&self, duration: TimeCode) -> bool {
        let (_, end) = self.playback_range(duration);
        self.current_time.as_secs() >= end.as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_defaults() {
        let ps = PlaybackState::new();
        assert_eq!(ps.mode, PlaybackMode::Stopped);
        assert_eq!(ps.current_time.as_secs(), 0.0);
        assert!(ps.in_point.is_none());
        assert!(ps.out_point.is_none());
        assert!(!ps.loop_enabled);
        assert!((ps.playback_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn play_pause_stop_cycle() {
        let mut ps = PlaybackState::new();
        ps.play();
        assert_eq!(ps.mode, PlaybackMode::Playing);
        assert!(ps.is_active());

        ps.pause();
        assert_eq!(ps.mode, PlaybackMode::Paused);
        assert!(!ps.is_active());

        ps.seek(TimeCode::from_secs(5.0));
        assert_eq!(ps.current_time.as_secs(), 5.0);

        ps.stop();
        assert_eq!(ps.mode, PlaybackMode::Stopped);
        assert_eq!(ps.current_time.as_secs(), 0.0);
    }

    #[test]
    fn stop_resets_to_in_point() {
        let mut ps = PlaybackState::new();
        ps.set_in_point(TimeCode::from_secs(2.0));
        ps.play();
        ps.seek(TimeCode::from_secs(10.0));
        ps.stop();
        assert_eq!(ps.current_time.as_secs(), 2.0);
    }

    #[test]
    fn toggle_play_pause() {
        let mut ps = PlaybackState::new();
        assert_eq!(ps.mode, PlaybackMode::Stopped);

        ps.toggle_play_pause();
        assert_eq!(ps.mode, PlaybackMode::Playing);

        ps.toggle_play_pause();
        assert_eq!(ps.mode, PlaybackMode::Paused);

        ps.toggle_play_pause();
        assert_eq!(ps.mode, PlaybackMode::Playing);
    }

    #[test]
    fn toggle_from_scrubbing_starts_play() {
        let mut ps = PlaybackState::new();
        ps.start_scrub(TimeCode::from_secs(3.0));
        assert_eq!(ps.mode, PlaybackMode::Scrubbing);

        ps.toggle_play_pause();
        assert_eq!(ps.mode, PlaybackMode::Playing);
    }

    #[test]
    fn seek_from_stopped_transitions_to_paused() {
        let mut ps = PlaybackState::new();
        assert_eq!(ps.mode, PlaybackMode::Stopped);
        ps.seek(TimeCode::from_secs(5.0));
        assert_eq!(ps.mode, PlaybackMode::Paused);
        assert_eq!(ps.current_time.as_secs(), 5.0);
    }

    #[test]
    fn seek_from_playing_stays_playing() {
        let mut ps = PlaybackState::new();
        ps.play();
        ps.seek(TimeCode::from_secs(5.0));
        assert_eq!(ps.mode, PlaybackMode::Playing);
        assert_eq!(ps.current_time.as_secs(), 5.0);
    }

    #[test]
    fn scrub_lifecycle() {
        let mut ps = PlaybackState::new();
        ps.start_scrub(TimeCode::from_secs(1.0));
        assert_eq!(ps.mode, PlaybackMode::Scrubbing);
        assert_eq!(ps.current_time.as_secs(), 1.0);

        ps.scrub_to(TimeCode::from_secs(3.0));
        assert_eq!(ps.current_time.as_secs(), 3.0);

        ps.end_scrub();
        assert_eq!(ps.mode, PlaybackMode::Paused);
        assert_eq!(ps.current_time.as_secs(), 3.0);
    }

    #[test]
    fn scrub_to_only_works_in_scrubbing_mode() {
        let mut ps = PlaybackState::new();
        ps.seek(TimeCode::from_secs(1.0));
        ps.scrub_to(TimeCode::from_secs(5.0));
        // Should not change because not in scrubbing mode
        assert_eq!(ps.current_time.as_secs(), 1.0);
    }

    #[test]
    fn in_out_points() {
        let mut ps = PlaybackState::new();
        ps.set_in_point(TimeCode::from_secs(2.0));
        ps.set_out_point(TimeCode::from_secs(8.0));

        let (start, end) = ps.playback_range(TimeCode::from_secs(30.0));
        assert_eq!(start.as_secs(), 2.0);
        assert_eq!(end.as_secs(), 8.0);
    }

    #[test]
    fn playback_range_without_in_out() {
        let ps = PlaybackState::new();
        let duration = TimeCode::from_secs(60.0);
        let (start, end) = ps.playback_range(duration);
        assert_eq!(start.as_secs(), 0.0);
        assert_eq!(end.as_secs(), 60.0);
    }

    #[test]
    fn clear_in_out_points() {
        let mut ps = PlaybackState::new();
        ps.set_in_point(TimeCode::from_secs(2.0));
        ps.set_out_point(TimeCode::from_secs(8.0));
        ps.clear_in_out();
        assert!(ps.in_point.is_none());
        assert!(ps.out_point.is_none());
    }

    #[test]
    fn loop_toggle() {
        let mut ps = PlaybackState::new();
        assert!(!ps.loop_enabled);
        ps.set_loop(true);
        assert!(ps.loop_enabled);
        ps.set_loop(false);
        assert!(!ps.loop_enabled);
    }

    #[test]
    fn rate_clamping() {
        let mut ps = PlaybackState::new();

        ps.set_rate(2.0);
        assert!((ps.playback_rate - 2.0).abs() < f64::EPSILON);

        ps.set_rate(0.01);
        assert!((ps.playback_rate - 0.1).abs() < f64::EPSILON);

        ps.set_rate(100.0);
        assert!((ps.playback_rate - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn is_at_end() {
        let mut ps = PlaybackState::new();
        let duration = TimeCode::from_secs(10.0);

        assert!(!ps.is_at_end(duration));

        ps.seek(TimeCode::from_secs(10.0));
        assert!(ps.is_at_end(duration));

        ps.seek(TimeCode::from_secs(11.0));
        assert!(ps.is_at_end(duration));
    }

    #[test]
    fn is_at_end_with_out_point() {
        let mut ps = PlaybackState::new();
        ps.set_out_point(TimeCode::from_secs(5.0));
        let duration = TimeCode::from_secs(30.0);

        ps.seek(TimeCode::from_secs(4.9));
        assert!(!ps.is_at_end(duration));

        ps.seek(TimeCode::from_secs(5.0));
        assert!(ps.is_at_end(duration));
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut ps = PlaybackState::new();
        ps.play();
        ps.seek(TimeCode::from_secs(5.5));
        ps.set_in_point(TimeCode::from_secs(1.0));
        ps.set_out_point(TimeCode::from_secs(20.0));
        ps.set_loop(true);
        ps.set_rate(0.5);

        let json = serde_json::to_string(&ps).unwrap();
        let restored: PlaybackState = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.mode, PlaybackMode::Playing);
        assert_eq!(restored.current_time.as_secs(), 5.5);
        assert_eq!(restored.in_point.unwrap().as_secs(), 1.0);
        assert_eq!(restored.out_point.unwrap().as_secs(), 20.0);
        assert!(restored.loop_enabled);
        assert!((restored.playback_rate - 0.5).abs() < f64::EPSILON);
    }
}
