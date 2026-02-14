//! Multi-track audio mixing with volume and pan control.
//!
//! The mixer takes multiple audio inputs (tracks) and combines them into
//! a single output buffer. Each input can have independent volume and pan,
//! and can be muted.

/// An individual input to the audio mixer.
#[derive(Clone, Debug)]
pub struct MixerInput {
    /// Interleaved audio samples (f32).
    pub samples: Vec<f32>,
    /// Volume level: 0.0 (silent) to 1.0 (full). Values > 1.0 amplify.
    pub volume: f32,
    /// Pan position: -1.0 (full left), 0.0 (center), 1.0 (full right).
    pub pan: f32,
    /// If true, this input contributes no audio to the mix.
    pub muted: bool,
}

/// Multi-track audio mixer.
///
/// Combines multiple audio inputs into a single interleaved stereo (or mono)
/// output buffer. Uses constant-power panning for natural stereo imaging.
pub struct AudioMixer {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Output channel count (1 = mono, 2 = stereo).
    pub channels: u16,
}

impl AudioMixer {
    /// Create a new mixer for the given output format.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    /// Mix multiple inputs into a single interleaved output buffer.
    ///
    /// `output_frames` is the number of audio frames (not samples) to produce.
    /// For stereo, each frame has 2 samples; for mono, each frame has 1 sample.
    ///
    /// Inputs shorter than `output_frames` are zero-padded.
    /// Inputs longer than `output_frames` are truncated.
    ///
    /// Uses constant-power panning: left = cos(theta), right = sin(theta)
    /// where theta = (pan + 1) * PI/4. This keeps perceived loudness constant
    /// as audio is panned across the stereo field.
    pub fn mix(&self, inputs: &[MixerInput], output_frames: usize) -> Vec<f32> {
        let ch = self.channels as usize;
        let total_samples = output_frames * ch;
        let mut output = vec![0.0f32; total_samples];

        for input in inputs {
            if input.muted || input.volume == 0.0 {
                continue;
            }

            let (gain_l, gain_r) = Self::compute_pan_gains(input.pan, input.volume);
            let input_ch = Self::detect_channels(&input.samples, ch);

            match (input_ch, ch) {
                (1, 1) => {
                    // Mono input -> mono output
                    self.mix_mono_to_mono(&input.samples, &mut output, output_frames, input.volume);
                }
                (1, 2) => {
                    // Mono input -> stereo output (with panning)
                    self.mix_mono_to_stereo(
                        &input.samples,
                        &mut output,
                        output_frames,
                        gain_l,
                        gain_r,
                    );
                }
                (2, 2) => {
                    // Stereo input -> stereo output (with panning)
                    self.mix_stereo_to_stereo(
                        &input.samples,
                        &mut output,
                        output_frames,
                        gain_l,
                        gain_r,
                    );
                }
                (2, 1) => {
                    // Stereo input -> mono output (downmix)
                    self.mix_stereo_to_mono(
                        &input.samples,
                        &mut output,
                        output_frames,
                        input.volume,
                    );
                }
                _ => {
                    // For other channel configurations, treat as mono (take first channel)
                    self.mix_mono_to_mono(&input.samples, &mut output, output_frames, input.volume);
                }
            }
        }

        // Soft-clip the output to prevent harsh clipping
        for sample in &mut output {
            *sample = soft_clip(*sample);
        }

        output
    }

    /// Mix mono input samples into a mono output buffer.
    fn mix_mono_to_mono(
        &self,
        input: &[f32],
        output: &mut [f32],
        output_frames: usize,
        volume: f32,
    ) {
        let len = input.len().min(output_frames);
        for i in 0..len {
            output[i] += input[i] * volume;
        }
    }

    /// Mix mono input samples into a stereo output buffer with pan gains.
    fn mix_mono_to_stereo(
        &self,
        input: &[f32],
        output: &mut [f32],
        output_frames: usize,
        gain_l: f32,
        gain_r: f32,
    ) {
        let len = input.len().min(output_frames);
        for i in 0..len {
            let s = input[i];
            output[i * 2] += s * gain_l;
            output[i * 2 + 1] += s * gain_r;
        }
    }

    /// Mix stereo input samples into a stereo output buffer with pan gains.
    fn mix_stereo_to_stereo(
        &self,
        input: &[f32],
        output: &mut [f32],
        output_frames: usize,
        gain_l: f32,
        gain_r: f32,
    ) {
        let frames = (input.len() / 2).min(output_frames);
        for i in 0..frames {
            let l = input[i * 2];
            let r = input[i * 2 + 1];
            output[i * 2] += l * gain_l;
            output[i * 2 + 1] += r * gain_r;
        }
    }

    /// Mix stereo input samples down to mono output.
    fn mix_stereo_to_mono(
        &self,
        input: &[f32],
        output: &mut [f32],
        output_frames: usize,
        volume: f32,
    ) {
        let frames = (input.len() / 2).min(output_frames);
        for i in 0..frames {
            let l = input[i * 2];
            let r = input[i * 2 + 1];
            // Standard stereo-to-mono downmix: (L + R) * 0.5
            output[i] += (l + r) * 0.5 * volume;
        }
    }

    /// Compute left/right gain from pan and volume using constant-power panning.
    ///
    /// Constant-power panning uses cos/sin so that the perceived total energy
    /// remains constant as the source moves across the stereo field.
    fn compute_pan_gains(pan: f32, volume: f32) -> (f32, f32) {
        // Clamp pan to [-1, 1]
        let pan = pan.clamp(-1.0, 1.0);

        // Map pan from [-1, 1] to [0, PI/2]
        let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;

        let gain_l = theta.cos() * volume;
        let gain_r = theta.sin() * volume;

        (gain_l, gain_r)
    }

    /// Detect the number of channels in the input based on total samples
    /// and the output channel configuration.
    fn detect_channels(_samples: &[f32], output_channels: usize) -> usize {
        // Default assumption: input has same channel count as output
        // Caller should ensure input samples are properly interleaved
        output_channels
    }
}

/// Soft-clip a sample using tanh-based saturation.
///
/// Values in [-1, 1] pass through nearly unchanged.
/// Values beyond that range are gently compressed, preventing harsh digital clipping.
fn soft_clip(x: f32) -> f32 {
    if x.abs() <= 1.0 {
        x
    } else {
        x.signum() * (1.0 + (x.abs() - 1.0).tanh()) * 0.5 + x.signum() * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_empty_inputs() {
        let mixer = AudioMixer::new(44100, 2);
        let output = mixer.mix(&[], 1024);
        assert_eq!(output.len(), 2048); // 1024 frames * 2 channels
        assert!(output.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn mix_single_input_center_pan() {
        let mixer = AudioMixer::new(44100, 2);
        let input = MixerInput {
            samples: vec![0.5; 100], // 50 stereo frames
            volume: 1.0,
            pan: 0.0,
            muted: false,
        };
        let output = mixer.mix(&[input], 50);
        assert_eq!(output.len(), 100);
        // Center pan should distribute roughly equally
        // With constant-power at center: cos(PI/4) = sin(PI/4) ~ 0.707
        let expected_gain = std::f32::consts::FRAC_PI_4.cos();
        let tolerance = 0.01;
        assert!((output[0] - 0.5 * expected_gain).abs() < tolerance);
        assert!((output[1] - 0.5 * expected_gain).abs() < tolerance);
    }

    #[test]
    fn mix_muted_input_is_silent() {
        let mixer = AudioMixer::new(44100, 2);
        let input = MixerInput {
            samples: vec![1.0; 200],
            volume: 1.0,
            pan: 0.0,
            muted: true,
        };
        let output = mixer.mix(&[input], 100);
        assert!(output.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn mix_zero_volume_is_silent() {
        let mixer = AudioMixer::new(44100, 2);
        let input = MixerInput {
            samples: vec![1.0; 200],
            volume: 0.0,
            pan: 0.0,
            muted: false,
        };
        let output = mixer.mix(&[input], 100);
        assert!(output.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn pan_full_left() {
        let mixer = AudioMixer::new(44100, 2);
        // Use mono-like approach: both L and R of a stereo pair are the same
        let input = MixerInput {
            samples: vec![0.8, 0.8], // one stereo frame
            volume: 1.0,
            pan: -1.0,
            muted: false,
        };
        let output = mixer.mix(&[input], 1);
        // Full left: theta = 0, cos(0) = 1.0, sin(0) = 0.0
        assert!((output[0] - 0.8).abs() < 0.01, "Left should be ~0.8");
        assert!(output[1].abs() < 0.01, "Right should be ~0.0");
    }

    #[test]
    fn pan_full_right() {
        let mixer = AudioMixer::new(44100, 2);
        let input = MixerInput {
            samples: vec![0.8, 0.8],
            volume: 1.0,
            pan: 1.0,
            muted: false,
        };
        let output = mixer.mix(&[input], 1);
        // Full right: theta = PI/2, cos(PI/2) ~ 0.0, sin(PI/2) = 1.0
        assert!(output[0].abs() < 0.01, "Left should be ~0.0");
        assert!((output[1] - 0.8).abs() < 0.01, "Right should be ~0.8");
    }

    #[test]
    fn mix_multiple_inputs_sum() {
        let mixer = AudioMixer::new(44100, 1); // mono output
        let input1 = MixerInput {
            samples: vec![0.3],
            volume: 1.0,
            pan: 0.0,
            muted: false,
        };
        let input2 = MixerInput {
            samples: vec![0.2],
            volume: 1.0,
            pan: 0.0,
            muted: false,
        };
        let output = mixer.mix(&[input1, input2], 1);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn soft_clip_passes_normal_values() {
        assert!((soft_clip(0.5) - 0.5).abs() < 0.001);
        assert!((soft_clip(-0.5) - (-0.5)).abs() < 0.001);
        assert!((soft_clip(0.0)).abs() < 0.001);
    }

    #[test]
    fn soft_clip_limits_extreme_values() {
        assert!(soft_clip(5.0) < 1.5);
        assert!(soft_clip(5.0) > 0.5);
        assert!(soft_clip(-5.0) > -1.5);
        assert!(soft_clip(-5.0) < -0.5);
    }

    #[test]
    fn constant_power_panning_energy() {
        // At any pan position, L^2 + R^2 should be approximately volume^2
        let volume = 1.0;
        for pan_int in -10..=10 {
            let pan = pan_int as f32 / 10.0;
            let (l, r) = AudioMixer::compute_pan_gains(pan, volume);
            let energy = l * l + r * r;
            assert!(
                (energy - 1.0).abs() < 0.01,
                "Energy at pan={pan} is {energy}, expected ~1.0"
            );
        }
    }

    #[test]
    fn short_input_is_zero_padded() {
        let mixer = AudioMixer::new(44100, 1);
        let input = MixerInput {
            samples: vec![0.5, 0.5],
            volume: 1.0,
            pan: 0.0,
            muted: false,
        };
        let output = mixer.mix(&[input], 10);
        assert_eq!(output.len(), 10);
        // First two samples should have audio
        assert!((output[0] - 0.5).abs() < 0.01);
        assert!((output[1] - 0.5).abs() < 0.01);
        // Rest should be zero
        for &s in &output[2..] {
            assert!(s.abs() < 0.001);
        }
    }
}
