//! Sample rate conversion using linear interpolation.
//!
//! Provides a simple but effective resampler for converting audio between
//! different sample rates (e.g., 48kHz source to 44.1kHz output).
//! Uses linear interpolation which is good enough for preview playback.
//! For export/mastering quality, a higher-order resampler should be used.

/// Linear interpolation resampler for multi-channel audio.
///
/// Converts interleaved audio from one sample rate to another.
/// Maintains internal state (fractional position) across calls to
/// [`process`](Self::process) so it can be used in a streaming fashion.
pub struct Resampler {
    /// Source sample rate in Hz.
    from_rate: u32,
    /// Target sample rate in Hz.
    to_rate: u32,
    /// Number of audio channels (interleaved).
    channels: u16,
    /// Ratio: to_rate / from_rate.
    ratio: f64,
    /// Fractional position carried across process() calls.
    fractional_pos: f64,
    /// Last sample per channel from previous chunk (for interpolation across boundaries).
    last_samples: Vec<f32>,
}

impl Resampler {
    /// Create a new resampler.
    ///
    /// # Arguments
    ///
    /// * `from_rate` - Source sample rate in Hz
    /// * `to_rate` - Target sample rate in Hz
    /// * `channels` - Number of interleaved channels
    pub fn new(from_rate: u32, to_rate: u32, channels: u16) -> Self {
        Self {
            from_rate,
            to_rate,
            channels,
            ratio: to_rate as f64 / from_rate as f64,
            fractional_pos: 0.0,
            last_samples: vec![0.0; channels as usize],
        }
    }

    /// Resample a chunk of interleaved audio.
    ///
    /// Input samples must be interleaved (L, R, L, R, ... for stereo).
    /// Returns resampled interleaved samples at the target rate.
    ///
    /// This method maintains state between calls, so consecutive chunks
    /// of audio can be resampled without gaps or clicks at boundaries.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let ch = self.channels as usize;

        if ch == 0 || input.is_empty() {
            return Vec::new();
        }

        // If rates are the same, just pass through
        if self.from_rate == self.to_rate {
            return input.to_vec();
        }

        let input_frames = input.len() / ch;
        if input_frames == 0 {
            return Vec::new();
        }

        // Estimate output size
        let estimated_output_frames =
            ((input_frames as f64 * self.ratio).ceil() as usize).max(1) + 1;
        let mut output = Vec::with_capacity(estimated_output_frames * ch);

        // Step through the input at the source rate, outputting at the target rate
        let step = 1.0 / self.ratio; // input frames per output frame
        let mut pos = self.fractional_pos;

        while pos < input_frames as f64 {
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;

            for c in 0..ch {
                let sample_a = if idx == 0 && pos < 1.0 {
                    // Use last sample from previous chunk for interpolation
                    self.last_samples[c]
                } else if idx < input_frames {
                    input[idx * ch + c]
                } else {
                    0.0
                };

                let sample_b = if idx + 1 < input_frames {
                    input[(idx + 1) * ch + c]
                } else if idx < input_frames {
                    input[idx * ch + c]
                } else {
                    0.0
                };

                // Linear interpolation
                let interpolated = sample_a + (sample_b - sample_a) * frac;
                output.push(interpolated);
            }

            pos += step;
        }

        // Save state for next call
        self.fractional_pos = pos - input_frames as f64;

        // Save last samples for cross-boundary interpolation
        if input_frames > 0 {
            for c in 0..ch {
                self.last_samples[c] = input[(input_frames - 1) * ch + c];
            }
        }

        output
    }

    /// Reset the resampler state.
    ///
    /// Call this when seeking or switching sources to prevent
    /// interpolation artifacts from stale data.
    pub fn reset(&mut self) {
        self.fractional_pos = 0.0;
        self.last_samples.fill(0.0);
    }

    /// Get the resampling ratio (target_rate / source_rate).
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Get the source sample rate.
    pub fn from_rate(&self) -> u32 {
        self.from_rate
    }

    /// Get the target sample rate.
    pub fn to_rate(&self) -> u32 {
        self.to_rate
    }
}

/// One-shot resample without maintaining state.
///
/// Useful for resampling a complete audio buffer in one call.
pub fn resample_buffer(input: &[f32], from_rate: u32, to_rate: u32, channels: u16) -> Vec<f32> {
    let mut resampler = Resampler::new(from_rate, to_rate, channels);
    resampler.process(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_rate_passthrough() {
        let mut resampler = Resampler::new(44100, 44100, 1);
        let input: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let output = resampler.process(&input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn upsample_doubles_approximately() {
        let mut resampler = Resampler::new(22050, 44100, 1);
        let input = vec![0.0; 100];
        let output = resampler.process(&input);
        // Output should be approximately double the input length
        let expected_approx = 200;
        assert!(
            (output.len() as i32 - expected_approx as i32).unsigned_abs() <= 2,
            "Expected ~{expected_approx} samples, got {}",
            output.len()
        );
    }

    #[test]
    fn downsample_halves_approximately() {
        let mut resampler = Resampler::new(44100, 22050, 1);
        let input = vec![0.0; 100];
        let output = resampler.process(&input);
        // Output should be approximately half the input length
        let expected_approx = 50;
        assert!(
            (output.len() as i32 - expected_approx as i32).unsigned_abs() <= 2,
            "Expected ~{expected_approx} samples, got {}",
            output.len()
        );
    }

    #[test]
    fn stereo_resampling() {
        let mut resampler = Resampler::new(44100, 44100, 2);
        // 4 stereo frames: [L0, R0, L1, R1, L2, R2, L3, R3]
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let output = resampler.process(&input);
        assert_eq!(output.len(), 8); // Same rate, same size
        assert!((output[0] - 0.1).abs() < 1e-5);
        assert!((output[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn interpolation_produces_smooth_output() {
        // Generate a sine wave at 22050 Hz, upsample to 44100 Hz
        let from_rate = 22050u32;
        let to_rate = 44100u32;
        let freq = 440.0; // 440 Hz sine wave
        let num_samples = 1000;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / from_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();

        let mut resampler = Resampler::new(from_rate, to_rate, 1);
        let output = resampler.process(&input);

        // Verify output is reasonable: no NaN, no Inf, values in [-1, 1]
        for &s in &output {
            assert!(s.is_finite(), "Sample is not finite: {s}");
            assert!(s.abs() <= 1.01, "Sample out of range: {s}");
        }

        // Output should be approximately double
        assert!(output.len() >= 1900 && output.len() <= 2100);
    }

    #[test]
    fn reset_clears_state() {
        let mut resampler = Resampler::new(44100, 48000, 1);
        let input = vec![0.5; 100];
        resampler.process(&input);

        resampler.reset();
        assert!((resampler.fractional_pos).abs() < 1e-9);
        assert!(resampler.last_samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn empty_input() {
        let mut resampler = Resampler::new(44100, 48000, 1);
        let output = resampler.process(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn resample_buffer_convenience() {
        let input = vec![0.0; 100];
        let output = resample_buffer(&input, 44100, 44100, 1);
        assert_eq!(output.len(), 100);
    }

    #[test]
    fn ratio_correct() {
        let resampler = Resampler::new(44100, 48000, 2);
        assert!((resampler.ratio() - 48000.0 / 44100.0).abs() < 1e-9);
    }

    #[test]
    fn common_rate_conversions() {
        // 48kHz -> 44.1kHz (common for video -> CD quality)
        let mut r = Resampler::new(48000, 44100, 1);
        let input = vec![0.0; 480];
        let output = r.process(&input);
        let expected = (480.0_f64 * 44100.0 / 48000.0).round() as usize;
        assert!(
            (output.len() as i32 - expected as i32).unsigned_abs() <= 2,
            "48k->44.1k: expected ~{expected}, got {}",
            output.len()
        );

        // 44.1kHz -> 48kHz
        let mut r = Resampler::new(44100, 48000, 1);
        let input = vec![0.0; 441];
        let output = r.process(&input);
        let expected = (441.0_f64 * 48000.0 / 44100.0).round() as usize;
        assert!(
            (output.len() as i32 - expected as i32).unsigned_abs() <= 2,
            "44.1k->48k: expected ~{expected}, got {}",
            output.len()
        );
    }
}
