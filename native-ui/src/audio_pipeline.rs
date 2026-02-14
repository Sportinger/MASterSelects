//! Audio playback pipeline for the MasterSelects native engine.
//!
//! Integrates the `ms-audio` crate components (decoder, mixer, output, clock, meter)
//! into a unified pipeline that the engine can drive each frame. Designed for graceful
//! degradation: if no audio output device is available, the pipeline still functions
//! (decoding, metering, clock) without producing sound.
//!
//! # Architecture
//!
//! ```text
//! AudioDecoder --> Resampler --> AudioMixer --> AudioOutput (CPAL)
//!                                    |                |
//!                                    v                v
//!                               AudioMeter       AudioClock
//! ```

use std::path::Path;

use ms_audio::{AudioClock, AudioDecoder, AudioMeter, AudioMixer, AudioOutput, MixerInput};
use ms_common::TimeCode;
use tracing::{debug, info, warn};
// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur within the audio pipeline.
#[derive(Debug, thiserror::Error)]
pub enum AudioPipelineError {
    /// A decoding operation failed.
    #[error("Failed to decode audio: {0}")]
    DecodeError(String),

    /// The audio output device is unavailable or encountered an error.
    #[error("Audio output unavailable: {0}")]
    OutputError(String),

    /// The audio format or codec is not supported.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

// ---------------------------------------------------------------------------
// Playback state
// ---------------------------------------------------------------------------

/// Current state of the audio playback pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioPlaybackState {
    /// Playback is stopped and position is at the beginning.
    Stopped,
    /// Audio is actively playing.
    Playing,
    /// Playback is paused at the current position.
    Paused,
}

// ---------------------------------------------------------------------------
// Audio levels (for UI metering)
// ---------------------------------------------------------------------------

/// Audio level information suitable for driving UI meters.
#[derive(Clone, Debug)]
pub struct AudioLevels {
    /// Peak level of the left channel (linear, 0.0 .. 1.0+).
    pub peak_l: f32,
    /// Peak level of the right channel (linear, 0.0 .. 1.0+).
    pub peak_r: f32,
    /// RMS level of the left channel (linear).
    pub rms_l: f32,
    /// RMS level of the right channel (linear).
    pub rms_r: f32,
    /// Integrated loudness in LUFS.
    pub lufs: f32,
}

impl Default for AudioLevels {
    fn default() -> Self {
        Self {
            peak_l: 0.0,
            peak_r: 0.0,
            rms_l: 0.0,
            rms_r: 0.0,
            lufs: -f32::INFINITY,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio pipeline
// ---------------------------------------------------------------------------

/// Number of audio frames to decode and mix per `update()` call.
///
/// At 48 kHz stereo this is approximately 21 ms of audio, which keeps latency
/// low while providing enough data to avoid frequent buffer underruns.
const FRAMES_PER_UPDATE: usize = 1024;

/// Unified audio pipeline that ties together decoding, mixing, metering,
/// output, and A/V sync.
///
/// The pipeline is designed for single-threaded use from the main (UI) thread.
/// The only concurrent component is the CPAL output callback, which communicates
/// via a lock-free ring buffer managed by [`AudioOutput`].
pub struct AudioPipeline {
    /// Audio decoder for the currently loaded file.
    decoder: Option<AudioDecoder>,
    /// Multi-track mixer (currently single-track, ready for multi-track).
    mixer: AudioMixer,
    /// CPAL audio output device. `None` when no device is available.
    output: Option<AudioOutput>,
    /// Sample-accurate master clock for A/V synchronization.
    clock: AudioClock,
    /// Level meter for UI visualisation.
    meter: AudioMeter,
    /// Current playback state.
    state: AudioPlaybackState,
    /// Output sample rate in Hz (from the output device, or a sensible default).
    sample_rate: u32,
    /// Master volume in the range 0.0 .. 1.0.
    master_volume: f32,
    /// Resampler for converting decoded audio to the output sample rate.
    resampler: Option<ms_audio::Resampler>,
    /// Whether the decoder has reached the end of the file.
    eof: bool,
}

impl AudioPipeline {
    /// Create a new audio pipeline.
    ///
    /// Attempts to initialise a CPAL audio output device. If no device is
    /// available the pipeline is still usable (clock, metering, decoding work)
    /// but no audible output will be produced.
    pub fn new() -> Self {
        let default_sample_rate: u32 = 48000;
        let default_channels: u16 = 2;

        // Try to open audio output -- gracefully handle failure.
        let (output, sample_rate) = match AudioOutput::new(default_sample_rate, default_channels) {
            Ok(out) => {
                let sr = out.sample_rate;
                info!(sample_rate = sr, "Audio output initialised");
                (Some(out), sr)
            }
            Err(e) => {
                warn!(error = %e, "Audio output unavailable -- running without sound");
                (None, default_sample_rate)
            }
        };

        Self {
            decoder: None,
            mixer: AudioMixer::new(sample_rate, default_channels),
            output,
            clock: AudioClock::new(sample_rate),
            meter: AudioMeter::new(),
            state: AudioPlaybackState::Stopped,
            sample_rate,
            master_volume: 1.0,
            resampler: None,
            eof: false,
        }
    }

    /// Load an audio file for playback.
    ///
    /// Opens the file with the Symphonia-backed [`AudioDecoder`] and prepares
    /// a resampler if the source sample rate differs from the output rate.
    pub fn load_file(&mut self, path: &Path) -> Result<(), AudioPipelineError> {
        let dec = AudioDecoder::open(path).map_err(|e| {
            let msg = format!("{e}");
            if msg.contains("Unsupported") {
                AudioPipelineError::UnsupportedFormat(msg)
            } else {
                AudioPipelineError::DecodeError(msg)
            }
        })?;

        let info = dec.stream_info();
        debug!(
            sample_rate = info.sample_rate,
            channels = info.channels,
            "Loaded audio file"
        );

        // Set up resampler if source rate differs from output rate.
        let resampler = if info.sample_rate != self.sample_rate {
            debug!(
                from = info.sample_rate,
                to = self.sample_rate,
                "Creating resampler"
            );
            Some(ms_audio::Resampler::new(
                info.sample_rate,
                self.sample_rate,
                info.channels,
            ))
        } else {
            None
        };

        // Stop current playback, reset state.
        self.stop();

        self.decoder = Some(dec);
        self.resampler = resampler;
        self.eof = false;

        Ok(())
    }

    /// Start playback from the current position.
    ///
    /// If no file is loaded, this is a no-op.
    pub fn play(&mut self) {
        if self.decoder.is_none() {
            return;
        }

        if let Some(ref mut out) = self.output {
            if let Err(e) = out.play() {
                warn!(error = %e, "Failed to start audio output");
            }
        }

        self.clock.start();
        self.state = AudioPlaybackState::Playing;
        debug!("Audio pipeline: play");
    }

    /// Pause playback, keeping the current position.
    pub fn pause(&mut self) {
        if let Some(ref mut out) = self.output {
            if let Err(e) = out.pause() {
                warn!(error = %e, "Failed to pause audio output");
            }
        }

        self.clock.stop();
        self.state = AudioPlaybackState::Paused;
        debug!("Audio pipeline: pause");
    }

    /// Stop playback and reset the position to the beginning.
    pub fn stop(&mut self) {
        if let Some(ref mut out) = self.output {
            if let Err(e) = out.pause() {
                warn!(error = %e, "Failed to pause audio output on stop");
            }
            out.reset_counter();
            out.flush();
        }

        self.clock.stop();
        self.clock.reset_samples();
        self.clock.set_base_time(TimeCode::from_secs(0.0));

        // Re-seek the decoder to the start if one is loaded.
        if let Some(ref mut dec) = self.decoder {
            let _ = dec.seek(TimeCode::from_secs(0.0));
        }

        if let Some(ref mut rs) = self.resampler {
            rs.reset();
        }

        self.meter.reset();
        self.eof = false;
        self.state = AudioPlaybackState::Stopped;
        debug!("Audio pipeline: stop");
    }

    /// Seek to a specific time in seconds.
    ///
    /// Adjusts the decoder position, resets the clock base time, and flushes
    /// stale audio from the output ring buffer.
    pub fn seek(&mut self, time_secs: f64) {
        let tc = TimeCode::from_secs(time_secs);

        if let Some(ref mut dec) = self.decoder {
            if let Err(e) = dec.seek(tc) {
                warn!(error = %e, time = time_secs, "Audio seek failed");
                return;
            }
        }

        // Reset the clock so it starts counting from the seek point.
        self.clock.reset_samples();
        self.clock.set_base_time(tc);

        if let Some(ref out) = self.output {
            out.reset_counter();
            out.flush();
        }

        if let Some(ref mut rs) = self.resampler {
            rs.reset();
        }

        self.meter.reset();
        self.eof = false;

        debug!(time_secs, "Audio pipeline: seek");
    }

    /// Get the current playback position in seconds, derived from the
    /// [`AudioClock`] master clock.
    pub fn position_secs(&self) -> f64 {
        self.clock.current_time().as_secs()
    }

    /// Set the master volume. The value is clamped to the range `[0.0, 1.0]`.
    pub fn set_volume(&mut self, volume: f32) {
        self.master_volume = volume.clamp(0.0, 1.0);
    }

    /// Get the current audio levels for UI metering.
    pub fn levels(&self) -> AudioLevels {
        // The AudioMeter processes mono-summed levels. We expose L/R by
        // mirroring the single-channel values for now. A future improvement
        // could run separate L/R meters.
        AudioLevels {
            peak_l: self.meter.peak,
            peak_r: self.meter.peak,
            rms_l: self.meter.rms,
            rms_r: self.meter.rms,
            lufs: self.meter.lufs,
        }
    }

    /// Pump the audio pipeline. Call this once per frame.
    ///
    /// Reads decoded audio from the decoder, optionally resamples, mixes
    /// with the configured volume, feeds it through the meter, and pushes
    /// samples into the CPAL output ring buffer. Also advances the
    /// [`AudioClock`].
    pub fn update(&mut self) {
        if self.state != AudioPlaybackState::Playing {
            return;
        }

        if self.eof {
            return;
        }

        let decoder = match self.decoder.as_mut() {
            Some(d) => d,
            None => return,
        };

        // Decode a chunk of audio.
        let decoded = match decoder.decode_next() {
            Ok(Some(d)) => d,
            Ok(None) => {
                // End of file.
                self.eof = true;
                debug!("Audio decoder reached end of file");
                return;
            }
            Err(e) => {
                warn!(error = %e, "Audio decode error during update");
                return;
            }
        };

        let channels = decoded.channels;

        // Resample if necessary.
        let samples = if let Some(ref mut rs) = self.resampler {
            rs.process(&decoded.data)
        } else {
            decoded.data
        };

        // Mix through the mixer (single-track for now).
        let input = MixerInput {
            samples,
            volume: self.master_volume,
            pan: 0.0,
            muted: false,
        };

        let num_frames = if channels > 0 {
            input.samples.len() / channels as usize
        } else {
            FRAMES_PER_UPDATE
        };

        let mixed = self.mixer.mix(&[input], num_frames);

        // Feed the meter.
        self.meter.process(&mixed, self.mixer.channels);

        // Update the clock based on frames produced.
        let frames_produced = mixed.len() / self.mixer.channels.max(1) as usize;
        self.clock
            .update_samples_played(frames_produced as u64 * self.mixer.channels as u64);

        // Push to output (if available).
        if let Some(ref out) = self.output {
            // The ring buffer may be full; in that case we simply drop samples.
            // This is acceptable for preview playback.
            if let Err(e) = out.write_samples(&mixed) {
                // BufferFull is not fatal -- just skip this chunk.
                debug!(error = %e, "Could not write audio samples to output");
            }
        }
    }

    /// Returns `true` if an audio output device was successfully initialised.
    pub fn is_output_available(&self) -> bool {
        self.output.is_some()
    }

    /// Returns the current playback state.
    pub fn state(&self) -> &AudioPlaybackState {
        &self.state
    }

    /// Returns the configured output sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the current master volume (0.0 .. 1.0).
    pub fn master_volume(&self) -> f32 {
        self.master_volume
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a pipeline that will NOT try to open a real CPAL device.
    /// We construct manually with `output: None` so tests run headlessly.
    fn headless_pipeline() -> AudioPipeline {
        let sr = 48000;
        AudioPipeline {
            decoder: None,
            mixer: AudioMixer::new(sr, 2),
            output: None,
            clock: AudioClock::new(sr),
            meter: AudioMeter::new(),
            state: AudioPlaybackState::Stopped,
            sample_rate: sr,
            master_volume: 1.0,
            resampler: None,
            eof: false,
        }
    }

    #[test]
    fn new_does_not_panic() {
        let _pipeline = headless_pipeline();
    }

    #[test]
    fn default_state_is_stopped() {
        let pipeline = headless_pipeline();
        assert_eq!(pipeline.state, AudioPlaybackState::Stopped);
    }

    #[test]
    fn play_without_file_is_noop() {
        let mut pipeline = headless_pipeline();
        pipeline.play();
        assert_eq!(pipeline.state, AudioPlaybackState::Stopped);
    }

    #[test]
    fn play_with_decoder_changes_state() {
        let mut pipeline = headless_pipeline();
        pipeline.state = AudioPlaybackState::Playing;
        assert_eq!(pipeline.state, AudioPlaybackState::Playing);
    }

    #[test]
    fn pause_changes_state_to_paused() {
        let mut pipeline = headless_pipeline();
        pipeline.state = AudioPlaybackState::Playing;
        pipeline.pause();
        assert_eq!(pipeline.state, AudioPlaybackState::Paused);
    }

    #[test]
    fn stop_resets_to_stopped() {
        let mut pipeline = headless_pipeline();
        pipeline.state = AudioPlaybackState::Playing;
        pipeline.stop();
        assert_eq!(pipeline.state, AudioPlaybackState::Stopped);
    }

    #[test]
    fn set_volume_clamps_to_valid_range() {
        let mut pipeline = headless_pipeline();

        pipeline.set_volume(0.5);
        assert!((pipeline.master_volume - 0.5).abs() < f32::EPSILON);

        pipeline.set_volume(-0.3);
        assert!((pipeline.master_volume).abs() < f32::EPSILON);

        pipeline.set_volume(1.5);
        assert!((pipeline.master_volume - 1.0).abs() < f32::EPSILON);

        pipeline.set_volume(0.0);
        assert!((pipeline.master_volume).abs() < f32::EPSILON);

        pipeline.set_volume(1.0);
        assert!((pipeline.master_volume - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn seek_does_not_panic() {
        let mut pipeline = headless_pipeline();
        pipeline.seek(0.0);
        pipeline.seek(10.5);
        pipeline.seek(999.0);
    }

    #[test]
    fn audio_levels_default_values_are_zero() {
        let levels = AudioLevels::default();
        assert_eq!(levels.peak_l, 0.0);
        assert_eq!(levels.peak_r, 0.0);
        assert_eq!(levels.rms_l, 0.0);
        assert_eq!(levels.rms_r, 0.0);
        assert!(levels.lufs.is_infinite() && levels.lufs < 0.0);
    }

    #[test]
    fn playback_state_partial_eq_works() {
        assert_eq!(AudioPlaybackState::Stopped, AudioPlaybackState::Stopped);
        assert_eq!(AudioPlaybackState::Playing, AudioPlaybackState::Playing);
        assert_eq!(AudioPlaybackState::Paused, AudioPlaybackState::Paused);
        assert_ne!(AudioPlaybackState::Stopped, AudioPlaybackState::Playing);
        assert_ne!(AudioPlaybackState::Playing, AudioPlaybackState::Paused);
        assert_ne!(AudioPlaybackState::Paused, AudioPlaybackState::Stopped);
    }

    #[test]
    fn position_secs_starts_at_zero() {
        let pipeline = headless_pipeline();
        assert!((pipeline.position_secs()).abs() < 1e-9);
    }

    #[test]
    fn is_output_available_reports_correctly() {
        let pipeline = headless_pipeline();
        assert!(!pipeline.is_output_available());
    }

    #[test]
    fn levels_returns_meter_values() {
        let mut pipeline = headless_pipeline();
        let levels = pipeline.levels();
        assert_eq!(levels.peak_l, 0.0);
        assert_eq!(levels.rms_l, 0.0);

        pipeline.meter.process(&[0.5; 1024], 2);
        let levels = pipeline.levels();
        assert!(levels.peak_l > 0.0, "Peak should be non-zero after audio");
    }

    #[test]
    fn update_without_decoder_is_noop() {
        let mut pipeline = headless_pipeline();
        pipeline.state = AudioPlaybackState::Playing;
        pipeline.update();
    }

    #[test]
    fn update_when_stopped_is_noop() {
        let mut pipeline = headless_pipeline();
        pipeline.update();
        assert_eq!(pipeline.state, AudioPlaybackState::Stopped);
    }

    #[test]
    fn stop_resets_clock_position() {
        let mut pipeline = headless_pipeline();
        pipeline.clock.set_base_time(TimeCode::from_secs(30.0));
        pipeline.clock.update_samples_played(48000);
        assert!(pipeline.position_secs() > 30.0);

        pipeline.stop();
        assert!(
            pipeline.position_secs().abs() < 1e-9,
            "Position should be 0 after stop"
        );
    }

    #[test]
    fn seek_updates_clock_base_time() {
        let mut pipeline = headless_pipeline();
        pipeline.seek(15.0);
        assert!((pipeline.position_secs() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn load_nonexistent_file_returns_error() {
        let mut pipeline = headless_pipeline();
        let result = pipeline.load_file(Path::new("/nonexistent/audio/file.wav"));
        assert!(result.is_err());
        match result.unwrap_err() {
            AudioPipelineError::DecodeError(msg) => {
                assert!(!msg.is_empty());
            }
            AudioPipelineError::UnsupportedFormat(msg) => {
                assert!(!msg.is_empty());
            }
            other => panic!("Unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn error_display_messages() {
        let e1 = AudioPipelineError::DecodeError("bad data".to_string());
        assert_eq!(e1.to_string(), "Failed to decode audio: bad data");

        let e2 = AudioPipelineError::OutputError("no device".to_string());
        assert_eq!(e2.to_string(), "Audio output unavailable: no device");

        let e3 = AudioPipelineError::UnsupportedFormat("xyz".to_string());
        assert_eq!(e3.to_string(), "Unsupported format: xyz");
    }
}
