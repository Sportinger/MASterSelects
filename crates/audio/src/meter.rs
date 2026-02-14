//! Audio metering: Peak, RMS, and LUFS measurement.
//!
//! Provides real-time audio level metering suitable for driving UI meters.
//! Implements:
//! - **Peak**: Maximum absolute sample value (instant).
//! - **RMS**: Root Mean Square level (short-term loudness).
//! - **LUFS**: Simplified Integrated Loudness Units Full Scale (ITU-R BS.1770-inspired).

/// Audio level meter that processes audio buffers and tracks
/// peak, RMS, and LUFS levels.
///
/// Call [`process`](Self::process) with each audio buffer and read the
/// resulting levels. The meter uses exponential smoothing so levels
/// decay naturally when audio stops.
#[derive(Clone, Debug)]
pub struct AudioMeter {
    /// Current peak level (0.0 to 1.0+).
    pub peak: f32,
    /// Current RMS level (0.0 to 1.0+).
    pub rms: f32,
    /// Integrated loudness in LUFS (negative dB scale, typically -70 to 0).
    pub lufs: f32,
    /// Running sum of squared samples for LUFS integration.
    lufs_sum: f64,
    /// Total number of samples processed for LUFS integration.
    lufs_count: u64,
    /// Peak decay rate per sample (exponential).
    peak_decay: f32,
    /// RMS smoothing coefficient (exponential moving average).
    rms_smoothing: f32,
}

impl AudioMeter {
    /// Number of samples over which the peak decays from 1.0 to ~0.001.
    /// At 44.1kHz, 44100 samples = 1 second of decay.
    const PEAK_DECAY_SAMPLES: f32 = 44100.0;

    /// RMS window size in samples (approx 50ms at 44.1kHz).
    const RMS_WINDOW: f32 = 2205.0;

    /// Create a new audio meter with zero levels.
    pub fn new() -> Self {
        // Peak decay: after PEAK_DECAY_SAMPLES, peak should be ~1% of original
        // decay^N = 0.01 => decay = 0.01^(1/N)
        let peak_decay = 0.01f32.powf(1.0 / Self::PEAK_DECAY_SAMPLES);

        // RMS smoothing: EMA coefficient for ~50ms window
        let rms_smoothing = 1.0 - (-1.0 / Self::RMS_WINDOW).exp();

        Self {
            peak: 0.0,
            rms: 0.0,
            lufs: -f32::INFINITY,
            lufs_sum: 0.0,
            lufs_count: 0,
            peak_decay,
            rms_smoothing,
        }
    }

    /// Process a buffer of interleaved audio samples.
    ///
    /// Updates peak, RMS, and LUFS levels. For stereo audio, processes
    /// all channels together (the meter shows the combined level).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f32 audio samples
    /// * `channels` - Number of channels (1 = mono, 2 = stereo, etc.)
    pub fn process(&mut self, samples: &[f32], channels: u16) {
        if samples.is_empty() || channels == 0 {
            return;
        }

        let ch = channels as usize;
        let num_frames = samples.len() / ch;

        let mut max_peak = 0.0f32;
        let mut rms_accum = self.rms * self.rms; // Current RMS^2 for EMA

        for frame_idx in 0..num_frames {
            // Mix to mono for metering (average of all channels)
            let mut mono_sum = 0.0f32;
            for c in 0..ch {
                let idx = frame_idx * ch + c;
                if idx < samples.len() {
                    mono_sum += samples[idx];
                }
            }
            let mono = mono_sum / ch as f32;
            let abs_mono = mono.abs();

            // Peak (track maximum)
            max_peak = max_peak.max(abs_mono);

            // RMS (exponential moving average of squared samples)
            let sq = mono * mono;
            rms_accum = rms_accum * (1.0 - self.rms_smoothing) + sq * self.rms_smoothing;

            // LUFS integration (sum of squares)
            self.lufs_sum += sq as f64;
            self.lufs_count += 1;
        }

        // Update peak with decay
        self.peak *= self.peak_decay.powi(num_frames as i32);
        if max_peak > self.peak {
            self.peak = max_peak;
        }

        // Update RMS
        self.rms = rms_accum.sqrt();

        // Update LUFS (integrated loudness)
        if self.lufs_count > 0 {
            let mean_square = self.lufs_sum / self.lufs_count as f64;
            if mean_square > 0.0 {
                // LUFS = -0.691 + 10 * log10(mean_square)
                // The -0.691 offset is from ITU-R BS.1770
                self.lufs = -0.691 + 10.0 * (mean_square as f32).log10();
            } else {
                self.lufs = -f32::INFINITY;
            }
        }
    }

    /// Reset all meter levels to zero/silence.
    pub fn reset(&mut self) {
        self.peak = 0.0;
        self.rms = 0.0;
        self.lufs = -f32::INFINITY;
        self.lufs_sum = 0.0;
        self.lufs_count = 0;
    }

    /// Get the peak level in decibels (dBFS).
    ///
    /// Returns `-inf` for silence, 0.0 for full scale.
    pub fn peak_db(&self) -> f32 {
        if self.peak > 0.0 {
            20.0 * self.peak.log10()
        } else {
            -f32::INFINITY
        }
    }

    /// Get the RMS level in decibels (dBFS).
    pub fn rms_db(&self) -> f32 {
        if self.rms > 0.0 {
            20.0 * self.rms.log10()
        } else {
            -f32::INFINITY
        }
    }

    /// Check if the audio is clipping (peak exceeds 1.0).
    pub fn is_clipping(&self) -> bool {
        self.peak > 1.0
    }
}

impl Default for AudioMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meter_initial_state() {
        let meter = AudioMeter::new();
        assert_eq!(meter.peak, 0.0);
        assert_eq!(meter.rms, 0.0);
        assert!(meter.lufs.is_infinite() && meter.lufs < 0.0);
    }

    #[test]
    fn meter_process_silence() {
        let mut meter = AudioMeter::new();
        meter.process(&[0.0; 1024], 1);
        assert_eq!(meter.peak, 0.0);
        assert!((meter.rms).abs() < 1e-6);
    }

    #[test]
    fn meter_process_full_scale_mono() {
        let mut meter = AudioMeter::new();
        // Full-scale signal
        meter.process(&[1.0; 4410], 1);
        assert!((meter.peak - 1.0).abs() < 0.01);
        // RMS of constant 1.0 should approach 1.0
        assert!(meter.rms > 0.9);
    }

    #[test]
    fn meter_process_half_scale() {
        let mut meter = AudioMeter::new();
        meter.process(&[0.5; 4410], 1);
        assert!((meter.peak - 0.5).abs() < 0.01);
        // RMS of constant 0.5 should approach 0.5
        assert!((meter.rms - 0.5).abs() < 0.1);
    }

    #[test]
    fn meter_peak_db_full_scale() {
        let mut meter = AudioMeter::new();
        meter.process(&[1.0; 1024], 1);
        assert!((meter.peak_db() - 0.0).abs() < 0.1);
    }

    #[test]
    fn meter_peak_db_half_scale() {
        let mut meter = AudioMeter::new();
        meter.process(&[0.5; 1024], 1);
        // 20 * log10(0.5) ~ -6.02 dB
        assert!((meter.peak_db() - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn meter_peak_db_silence() {
        let meter = AudioMeter::new();
        assert!(meter.peak_db().is_infinite());
    }

    #[test]
    fn meter_clipping_detection() {
        let mut meter = AudioMeter::new();
        assert!(!meter.is_clipping());

        meter.process(&[1.5; 1024], 1);
        assert!(meter.is_clipping());
    }

    #[test]
    fn meter_peak_decays() {
        let mut meter = AudioMeter::new();
        meter.process(&[1.0; 100], 1);
        let peak_after_signal = meter.peak;

        // Process silence — peak should decay
        meter.process(&[0.0; 44100], 1);
        assert!(meter.peak < peak_after_signal);
        assert!(meter.peak < 0.1, "Peak should have decayed significantly");
    }

    #[test]
    fn meter_stereo() {
        let mut meter = AudioMeter::new();
        // Stereo: L=0.8, R=0.4 -> mono=(0.8+0.4)/2=0.6
        let samples: Vec<f32> = std::iter::repeat([0.8f32, 0.4f32])
            .take(1000)
            .flatten()
            .collect();
        meter.process(&samples, 2);
        // Peak should be around 0.6 (mono mix)
        assert!((meter.peak - 0.6).abs() < 0.05);
    }

    #[test]
    fn meter_lufs_full_scale() {
        let mut meter = AudioMeter::new();
        meter.process(&[1.0; 44100], 1);
        // LUFS of constant 1.0: -0.691 + 10*log10(1.0) = -0.691
        assert!((meter.lufs - (-0.691)).abs() < 0.1);
    }

    #[test]
    fn meter_reset() {
        let mut meter = AudioMeter::new();
        meter.process(&[0.8; 4410], 1);
        assert!(meter.peak > 0.0);

        meter.reset();
        assert_eq!(meter.peak, 0.0);
        assert_eq!(meter.rms, 0.0);
        assert!(meter.lufs.is_infinite());
    }

    #[test]
    fn meter_default_trait() {
        let meter = AudioMeter::default();
        assert_eq!(meter.peak, 0.0);
    }

    #[test]
    fn meter_empty_input() {
        let mut meter = AudioMeter::new();
        meter.process(&[], 1);
        assert_eq!(meter.peak, 0.0);
    }
}
