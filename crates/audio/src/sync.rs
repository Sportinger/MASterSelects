//! Audio-video synchronization using audio as master clock.
//!
//! In professional video editing, audio is the authoritative time source
//! because audio glitches (dropouts, pops) are far more perceptible than
//! occasional dropped video frames. The [`AudioClock`] tracks how many
//! samples have been played by the audio output and converts that to
//! a precise time position that the video renderer uses for sync.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use ms_common::TimeCode;

/// Audio-based master clock for A/V synchronization.
///
/// Tracks the number of audio samples that have been consumed by the output
/// device and provides the authoritative timeline position. The video renderer
/// queries this clock to determine which frame to display.
///
/// Thread-safe: all operations use atomic types so the clock can be
/// updated from the audio thread and read from the render thread.
pub struct AudioClock {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Total samples played since last reset (atomic for lock-free access).
    samples_played: Arc<AtomicU64>,
    /// Base time offset added to the computed time (for seek support).
    /// Stored as f64 bits for atomic access.
    base_time_bits: Arc<AtomicU64>,
    /// Whether the clock is running.
    running: Arc<AtomicBool>,
}

impl AudioClock {
    /// Create a new audio clock for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            samples_played: Arc::new(AtomicU64::new(0)),
            base_time_bits: Arc::new(AtomicU64::new(0.0f64.to_bits())),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the current playback time based on samples played.
    ///
    /// This is the authoritative time source for the entire engine.
    /// The video renderer should display the frame whose PTS is closest
    /// to this value.
    pub fn current_time(&self) -> TimeCode {
        let samples = self.samples_played.load(Ordering::Relaxed);
        let base = f64::from_bits(self.base_time_bits.load(Ordering::Relaxed));
        let elapsed = samples as f64 / self.sample_rate as f64;
        TimeCode::from_secs(base + elapsed)
    }

    /// Reset the clock to zero (or to a specific base time via [`set_base_time`](Self::set_base_time)).
    pub fn reset(&mut self) {
        self.samples_played.store(0, Ordering::Relaxed);
        self.base_time_bits
            .store(0.0f64.to_bits(), Ordering::Relaxed);
    }

    /// Set a base time offset (used after seeking).
    ///
    /// After seeking to 30.0s, call `set_base_time(TimeCode::from_secs(30.0))`
    /// and then `reset_samples()` so the clock reports 30.0s + (samples_since_seek / rate).
    pub fn set_base_time(&self, time: TimeCode) {
        self.base_time_bits
            .store(time.as_secs().to_bits(), Ordering::Relaxed);
    }

    /// Reset the sample counter without changing the base time.
    pub fn reset_samples(&self) {
        self.samples_played.store(0, Ordering::Relaxed);
    }

    /// Update the number of samples that have been played.
    ///
    /// Called from the audio output thread after each buffer is consumed.
    /// Uses relaxed ordering because we only need eventual consistency --
    /// the video renderer polling this value a few microseconds late is fine.
    pub fn update_samples_played(&self, count: u64) {
        self.samples_played.fetch_add(count, Ordering::Relaxed);
    }

    /// Get the total number of samples played since the last reset.
    pub fn total_samples(&self) -> u64 {
        self.samples_played.load(Ordering::Relaxed)
    }

    /// Start the clock.
    pub fn start(&self) {
        self.running.store(true, Ordering::Relaxed);
    }

    /// Stop the clock.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Whether the clock is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get the sample rate this clock was configured with.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get a shared reference to the samples-played counter.
    ///
    /// This can be passed to the audio output callback for lock-free updates.
    pub fn samples_played_ref(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.samples_played)
    }
}

impl Clone for AudioClock {
    fn clone(&self) -> Self {
        Self {
            sample_rate: self.sample_rate,
            samples_played: Arc::clone(&self.samples_played),
            base_time_bits: Arc::clone(&self.base_time_bits),
            running: Arc::clone(&self.running),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_initial_time_is_zero() {
        let clock = AudioClock::new(44100);
        let time = clock.current_time();
        assert!((time.as_secs()).abs() < 1e-9);
    }

    #[test]
    fn clock_time_after_samples() {
        let clock = AudioClock::new(44100);
        // Playing 44100 samples at 44100 Hz = 1.0 second
        clock.update_samples_played(44100);
        let time = clock.current_time();
        assert!((time.as_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn clock_time_with_base_offset() {
        let clock = AudioClock::new(48000);
        clock.set_base_time(TimeCode::from_secs(10.0));
        // 48000 samples at 48000 Hz = 1 second
        clock.update_samples_played(48000);
        let time = clock.current_time();
        assert!((time.as_secs() - 11.0).abs() < 1e-9);
    }

    #[test]
    fn clock_reset() {
        let mut clock = AudioClock::new(44100);
        clock.update_samples_played(44100);
        clock.set_base_time(TimeCode::from_secs(5.0));

        clock.reset();

        let time = clock.current_time();
        assert!((time.as_secs()).abs() < 1e-9);
    }

    #[test]
    fn clock_reset_samples_keeps_base() {
        let clock = AudioClock::new(44100);
        clock.set_base_time(TimeCode::from_secs(5.0));
        clock.update_samples_played(44100);

        clock.reset_samples();

        let time = clock.current_time();
        assert!((time.as_secs() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn clock_start_stop() {
        let clock = AudioClock::new(44100);
        assert!(!clock.is_running());

        clock.start();
        assert!(clock.is_running());

        clock.stop();
        assert!(!clock.is_running());
    }

    #[test]
    fn clock_is_clone_and_shared() {
        let clock1 = AudioClock::new(44100);
        let clock2 = clock1.clone();

        clock1.update_samples_played(44100);

        // Both clocks see the same samples played (shared Arc)
        assert_eq!(clock2.total_samples(), 44100);
        assert!((clock2.current_time().as_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn clock_fractional_time() {
        let clock = AudioClock::new(48000);
        // 24000 samples at 48000 Hz = 0.5 seconds
        clock.update_samples_played(24000);
        let time = clock.current_time();
        assert!((time.as_secs() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn clock_accumulates_updates() {
        let clock = AudioClock::new(44100);
        clock.update_samples_played(22050);
        clock.update_samples_played(22050);
        assert_eq!(clock.total_samples(), 44100);
        assert!((clock.current_time().as_secs() - 1.0).abs() < 1e-9);
    }
}
