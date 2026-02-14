//! Waveform data generation for timeline UI display.
//!
//! Generates min/max peak pairs from raw audio samples at a configurable
//! resolution (samples-per-peak). The resulting [`WaveformData`] is compact
//! enough to store per-clip and fast enough to render in egui.

/// Pre-computed waveform peaks for timeline visualization.
///
/// Each peak is a `(min, max)` pair representing the amplitude extremes
/// within a bucket of `samples_per_peak` mono samples.
#[derive(Clone, Debug)]
pub struct WaveformData {
    /// Min/max peak pairs. Values are in [-1.0, 1.0].
    pub peaks: Vec<(f32, f32)>,
    /// The sample rate of the source audio.
    pub sample_rate: u32,
    /// How many mono samples are represented by each peak pair.
    pub samples_per_peak: u32,
}

impl WaveformData {
    /// Generate waveform peaks from interleaved audio samples.
    ///
    /// For multi-channel audio, this first mixes down to mono (average of channels)
    /// before computing peaks. Each peak bucket covers `samples_per_peak` mono samples
    /// and stores the minimum and maximum sample value within that range.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f32 audio samples
    /// * `channels` - Number of audio channels (1 = mono, 2 = stereo, etc.)
    /// * `samples_per_peak` - Number of mono samples per peak bucket. Lower values
    ///   give higher resolution but use more memory.
    pub fn generate(samples: &[f32], channels: u16, samples_per_peak: u32) -> Self {
        if samples.is_empty() || samples_per_peak == 0 {
            return Self {
                peaks: Vec::new(),
                sample_rate: 0,
                samples_per_peak,
            };
        }

        let ch = channels as usize;
        let total_frames = samples.len() / ch;
        let num_peaks = total_frames.div_ceil(samples_per_peak as usize);

        let mut peaks = Vec::with_capacity(num_peaks);

        for peak_idx in 0..num_peaks {
            let start_frame = peak_idx * samples_per_peak as usize;
            let end_frame = (start_frame + samples_per_peak as usize).min(total_frames);

            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for frame in start_frame..end_frame {
                // Mix down to mono: average all channels
                let mut mono_sum = 0.0f32;
                for c in 0..ch {
                    let idx = frame * ch + c;
                    if idx < samples.len() {
                        mono_sum += samples[idx];
                    }
                }
                let mono = mono_sum / ch as f32;

                min_val = min_val.min(mono);
                max_val = max_val.max(mono);
            }

            peaks.push((min_val, max_val));
        }

        Self {
            peaks,
            sample_rate: 0, // Caller should set this
            samples_per_peak,
        }
    }

    /// Generate waveform peaks with a known sample rate.
    pub fn generate_with_rate(
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        samples_per_peak: u32,
    ) -> Self {
        let mut waveform = Self::generate(samples, channels, samples_per_peak);
        waveform.sample_rate = sample_rate;
        waveform
    }

    /// Get a subset of peaks for a time range.
    ///
    /// `start` and `end` are normalized positions in `[0.0, 1.0]` where
    /// 0.0 is the beginning and 1.0 is the end of the waveform.
    ///
    /// Returns an empty slice if the range is invalid.
    pub fn get_range(&self, start: f32, end: f32) -> &[(f32, f32)] {
        if self.peaks.is_empty() || start >= end || start >= 1.0 || end <= 0.0 {
            return &[];
        }

        let start = start.clamp(0.0, 1.0);
        let end = end.clamp(0.0, 1.0);

        let start_idx = (start * self.peaks.len() as f32) as usize;
        let end_idx = ((end * self.peaks.len() as f32).ceil() as usize).min(self.peaks.len());

        if start_idx >= end_idx || start_idx >= self.peaks.len() {
            return &[];
        }

        &self.peaks[start_idx..end_idx]
    }

    /// Get the duration in seconds (requires sample_rate to be set).
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        let total_samples = self.peaks.len() as f64 * self.samples_per_peak as f64;
        total_samples / self.sample_rate as f64
    }

    /// Get the total number of peaks.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Whether the waveform has no data.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    /// Resample the waveform to a different number of peaks.
    ///
    /// Useful for rendering at different zoom levels without re-analyzing
    /// the original audio. Uses min/max aggregation for downsampling
    /// and linear interpolation for upsampling.
    pub fn resample(&self, target_peaks: usize) -> Vec<(f32, f32)> {
        if self.peaks.is_empty() || target_peaks == 0 {
            return Vec::new();
        }

        if target_peaks == self.peaks.len() {
            return self.peaks.clone();
        }

        let mut result = Vec::with_capacity(target_peaks);
        let ratio = self.peaks.len() as f64 / target_peaks as f64;

        for i in 0..target_peaks {
            let src_start = (i as f64 * ratio) as usize;
            let src_end = (((i + 1) as f64 * ratio).ceil() as usize).min(self.peaks.len());

            if src_start >= self.peaks.len() {
                result.push((0.0, 0.0));
                continue;
            }

            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for j in src_start..src_end {
                min_val = min_val.min(self.peaks[j].0);
                max_val = max_val.max(self.peaks[j].1);
            }

            result.push((min_val, max_val));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_empty() {
        let waveform = WaveformData::generate(&[], 1, 256);
        assert!(waveform.is_empty());
    }

    #[test]
    fn generate_mono_simple() {
        // 8 samples, 4 samples per peak = 2 peaks
        let samples = vec![0.0, 0.5, -0.3, 0.8, -0.1, 0.2, 0.6, -0.7];
        let waveform = WaveformData::generate(&samples, 1, 4);

        assert_eq!(waveform.len(), 2);

        // First bucket: [0.0, 0.5, -0.3, 0.8] -> min=-0.3, max=0.8
        assert!((waveform.peaks[0].0 - (-0.3)).abs() < 1e-6);
        assert!((waveform.peaks[0].1 - 0.8).abs() < 1e-6);

        // Second bucket: [-0.1, 0.2, 0.6, -0.7] -> min=-0.7, max=0.6
        assert!((waveform.peaks[1].0 - (-0.7)).abs() < 1e-6);
        assert!((waveform.peaks[1].1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn generate_stereo_mixdown() {
        // 4 stereo frames (8 interleaved samples), 2 samples per peak = 2 peaks
        // Frame 0: L=0.4, R=0.6 -> mono = 0.5
        // Frame 1: L=-0.2, R=-0.8 -> mono = -0.5
        // Frame 2: L=0.0, R=1.0 -> mono = 0.5
        // Frame 3: L=0.3, R=-0.3 -> mono = 0.0
        let samples = vec![0.4, 0.6, -0.2, -0.8, 0.0, 1.0, 0.3, -0.3];
        let waveform = WaveformData::generate(&samples, 2, 2);

        assert_eq!(waveform.len(), 2);
        // Bucket 0: mono [0.5, -0.5] -> min=-0.5, max=0.5
        assert!((waveform.peaks[0].0 - (-0.5)).abs() < 1e-6);
        assert!((waveform.peaks[0].1 - 0.5).abs() < 1e-6);
        // Bucket 1: mono [0.5, 0.0] -> min=0.0, max=0.5
        assert!((waveform.peaks[1].0 - 0.0).abs() < 1e-6);
        assert!((waveform.peaks[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn generate_partial_last_bucket() {
        // 5 samples, 3 per peak = 2 peaks (last one has only 2 samples)
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let waveform = WaveformData::generate(&samples, 1, 3);

        assert_eq!(waveform.len(), 2);
        assert!((waveform.peaks[0].0 - 0.1).abs() < 1e-6);
        assert!((waveform.peaks[0].1 - 0.3).abs() < 1e-6);
        assert!((waveform.peaks[1].0 - 0.4).abs() < 1e-6);
        assert!((waveform.peaks[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn get_range_full() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let waveform = WaveformData::generate(&samples, 1, 1);
        let range = waveform.get_range(0.0, 1.0);
        assert_eq!(range.len(), 4);
    }

    #[test]
    fn get_range_first_half() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let waveform = WaveformData::generate(&samples, 1, 10);
        assert_eq!(waveform.len(), 10);

        let range = waveform.get_range(0.0, 0.5);
        assert_eq!(range.len(), 5);
    }

    #[test]
    fn get_range_invalid() {
        let waveform = WaveformData::generate(&[0.5; 100], 1, 10);
        assert!(waveform.get_range(0.8, 0.2).is_empty()); // start > end
        assert!(waveform.get_range(1.5, 2.0).is_empty()); // out of range
        assert!(waveform.get_range(-1.0, 0.0).is_empty()); // end <= 0
    }

    #[test]
    fn duration_with_sample_rate() {
        let waveform = WaveformData::generate_with_rate(&[0.0; 44100], 1, 44100, 441);
        assert!((waveform.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn resample_down() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();
        let waveform = WaveformData::generate(&samples, 1, 10);
        assert_eq!(waveform.len(), 100);

        let resampled = waveform.resample(10);
        assert_eq!(resampled.len(), 10);

        // Each resampled peak should encompass 10 original peaks
        for (min, max) in &resampled {
            assert!(*min <= *max);
        }
    }

    #[test]
    fn resample_same_size() {
        let waveform = WaveformData::generate(&[0.5; 100], 1, 10);
        let resampled = waveform.resample(waveform.len());
        assert_eq!(resampled.len(), waveform.len());
    }

    #[test]
    fn zero_samples_per_peak() {
        let waveform = WaveformData::generate(&[0.5; 100], 1, 0);
        assert!(waveform.is_empty());
    }
}
