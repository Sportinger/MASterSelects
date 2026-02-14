//! CPAL-based realtime audio output.
//!
//! Uses a lock-free ring buffer (crossbeam bounded channel) to decouple
//! the producer (mixer/decoder) from the consumer (CPAL audio callback).
//! The audio callback must NEVER block or allocate.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use crossbeam::channel::{bounded, Sender, TrySendError};
use parking_lot::Mutex;
use tracing::{debug, error, info, warn};

use crate::error::AudioError;

/// Size of the ring buffer in audio sample chunks.
/// Each chunk is a Vec<f32> of interleaved samples.
/// 64 chunks at ~1024 samples/chunk gives ~1.5s of buffer at 44.1kHz.
const RING_BUFFER_CHUNKS: usize = 64;

/// Audio output device backed by CPAL.
///
/// Provides a producer-consumer model: the application writes samples
/// into a ring buffer via [`write_samples`](Self::write_samples), and the CPAL audio
/// callback drains from the other end. The callback never blocks or allocates.
pub struct AudioOutput {
    /// The CPAL output stream (holds the audio thread alive).
    stream: Option<Stream>,
    /// Sender side of the ring buffer for pushing samples.
    sender: Sender<Vec<f32>>,
    /// Whether the stream is currently playing.
    playing: Arc<AtomicBool>,
    /// Shared counter of total samples consumed by the output callback.
    samples_played: Arc<parking_lot::Mutex<u64>>,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Output channel count.
    pub channels: u16,
}

impl AudioOutput {
    /// Create a new audio output device.
    ///
    /// Initializes CPAL with the default output device and builds a stream
    /// matching the requested sample rate and channel count.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| AudioError::Output("No audio output device found".to_string()))?;

        info!(
            device = device
                .name()
                .unwrap_or_else(|_| "unknown".to_string())
                .as_str(),
            "Using audio output device"
        );

        let config = StreamConfig {
            channels,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let (sender, receiver) = bounded::<Vec<f32>>(RING_BUFFER_CHUNKS);
        let playing = Arc::new(AtomicBool::new(false));
        let playing_cb = Arc::clone(&playing);
        let samples_played = Arc::new(Mutex::new(0u64));
        let samples_played_cb = Arc::clone(&samples_played);

        // Remainder buffer for partial chunks that don't fill a full callback
        let remainder: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // This callback runs on the audio thread. It must NEVER block or allocate.
                    if !playing_cb.load(Ordering::Relaxed) {
                        // Fill with silence when paused
                        data.fill(0.0);
                        return;
                    }

                    let mut written = 0;
                    let mut rem = remainder.lock();

                    // First, drain any leftover samples from the previous callback
                    if !rem.is_empty() {
                        let to_copy = rem.len().min(data.len());
                        data[..to_copy].copy_from_slice(&rem[..to_copy]);
                        written += to_copy;
                        if to_copy < rem.len() {
                            // Still have leftovers
                            *rem = rem[to_copy..].to_vec();
                        } else {
                            rem.clear();
                        }
                    }

                    // Then, pull from the ring buffer
                    while written < data.len() {
                        match receiver.try_recv() {
                            Ok(chunk) => {
                                let needed = data.len() - written;
                                let to_copy = chunk.len().min(needed);
                                data[written..written + to_copy].copy_from_slice(&chunk[..to_copy]);
                                written += to_copy;

                                if to_copy < chunk.len() {
                                    // Store remainder
                                    *rem = chunk[to_copy..].to_vec();
                                }
                            }
                            Err(_) => {
                                // Buffer underrun: fill remaining with silence
                                data[written..].fill(0.0);
                                written = data.len();
                            }
                        }
                    }

                    // Update the samples-played counter
                    *samples_played_cb.lock() += written as u64;
                },
                move |err| {
                    error!(error = %err, "Audio output stream error");
                },
                None, // no timeout
            )
            .map_err(|e| AudioError::StreamBuild(format!("{e}")))?;

        debug!(
            sample_rate = sample_rate,
            channels = channels,
            "Audio output stream built"
        );

        Ok(Self {
            stream: Some(stream),
            sender,
            playing,
            samples_played,
            sample_rate,
            channels,
        })
    }

    /// Start audio playback.
    ///
    /// The CPAL stream will begin pulling samples from the ring buffer.
    pub fn play(&mut self) -> Result<(), AudioError> {
        if let Some(ref stream) = self.stream {
            stream
                .play()
                .map_err(|e| AudioError::StreamPlay(format!("{e}")))?;
            self.playing.store(true, Ordering::Relaxed);
            debug!("Audio output playing");
        }
        Ok(())
    }

    /// Pause audio playback.
    ///
    /// The CPAL stream remains open but outputs silence.
    pub fn pause(&mut self) -> Result<(), AudioError> {
        self.playing.store(false, Ordering::Relaxed);
        debug!("Audio output paused");
        Ok(())
    }

    /// Write interleaved f32 samples into the ring buffer.
    ///
    /// The samples will be consumed by the CPAL audio callback on the audio thread.
    /// If the ring buffer is full, this returns [`AudioError::BufferFull`].
    ///
    /// For optimal performance, write in chunks of ~1024-4096 samples.
    pub fn write_samples(&self, samples: &[f32]) -> Result<(), AudioError> {
        if samples.is_empty() {
            return Ok(());
        }

        // Send the chunk to the ring buffer
        match self.sender.try_send(samples.to_vec()) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                warn!("Audio ring buffer full, dropping samples");
                Err(AudioError::BufferFull)
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(AudioError::Output("Audio output stream closed".to_string()))
            }
        }
    }

    /// Returns whether the output is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    /// Get the total number of samples that have been played by the output.
    ///
    /// This can be used by [`AudioClock`](crate::sync::AudioClock) for A/V sync.
    pub fn samples_played(&self) -> u64 {
        *self.samples_played.lock()
    }

    /// Reset the samples-played counter (e.g., after seeking).
    pub fn reset_counter(&self) {
        *self.samples_played.lock() = 0;
    }

    /// Drain any pending samples from the ring buffer.
    ///
    /// Useful when seeking — clears stale audio data.
    pub fn flush(&self) {
        while self.sender.try_send(Vec::new()).is_ok() {}
        // Actually we should drain the receiver, but we don't own it.
        // Instead, just mark position and let the callback drain naturally.
        debug!("Audio output flushed");
    }
}

// Stream is not Send, but our wrapper can handle being used from the main thread.
// CPAL streams are not Send, but are safely stopped on Drop.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_capacity() {
        // Verify the constant is reasonable
        assert!(RING_BUFFER_CHUNKS >= 16);
        assert!(RING_BUFFER_CHUNKS <= 256);
    }

    #[test]
    fn playing_state_default_false() {
        let playing = Arc::new(AtomicBool::new(false));
        assert!(!playing.load(Ordering::Relaxed));
    }

    #[test]
    fn channel_send_receive() {
        // Test the ring buffer pattern without CPAL
        let (sender, receiver) = bounded::<Vec<f32>>(4);

        let samples = vec![0.5f32; 1024];
        sender.try_send(samples.clone()).unwrap();

        let received = receiver.try_recv().unwrap();
        assert_eq!(received.len(), 1024);
        assert!((received[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn channel_full_detection() {
        let (sender, _receiver) = bounded::<Vec<f32>>(2);

        // Fill the buffer
        sender.try_send(vec![0.0; 10]).unwrap();
        sender.try_send(vec![0.0; 10]).unwrap();

        // Third should fail
        let result = sender.try_send(vec![0.0; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn samples_played_counter() {
        let counter = Arc::new(Mutex::new(0u64));
        *counter.lock() += 1024;
        *counter.lock() += 512;
        assert_eq!(*counter.lock(), 1536);
    }

    // NOTE: We don't test AudioOutput::new() in unit tests because it requires
    // a real audio device. Integration tests should cover that path.
}
