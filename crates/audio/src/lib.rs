//! `ms-audio` — Audio decoding, mixing, output, and metering for the MasterSelects native engine.
//!
//! This crate handles all audio functionality:
//!
//! - **Decoding**: Symphonia-based decode of AAC, MP3, FLAC, WAV, Opus, Vorbis
//! - **Mixing**: Multi-track mixing with volume, pan, and constant-power panning
//! - **Output**: CPAL-based realtime audio output with lock-free ring buffer
//! - **Sync**: Audio-as-master-clock for A/V synchronization
//! - **Waveform**: Peak data generation for timeline UI visualization
//! - **Resampler**: Linear interpolation sample rate conversion
//! - **Metering**: Peak, RMS, and LUFS level measurement
//!
//! # Architecture
//!
//! ```text
//! AudioDecoder -> Resampler -> AudioMixer -> AudioOutput (CPAL)
//!       |                          |               |
//!       v                          v               v
//! WaveformData                AudioMeter       AudioClock
//! ```
//!
//! The decoder reads compressed audio via Symphonia. If the source sample rate
//! differs from the output rate, the resampler converts it. Multiple decoded
//! tracks are combined by the mixer (with per-track volume/pan). The mixed
//! output feeds into CPAL via a lock-free ring buffer. The audio clock tracks
//! samples played for video sync, and the meter provides real-time levels.

pub mod decoder;
pub mod error;
pub mod meter;
pub mod mixer;
pub mod output;
pub mod resampler;
pub mod sync;
pub mod waveform;

// Re-export primary types at crate root for convenience
pub use decoder::{AudioDecoder, DecodedAudio};
pub use error::AudioError;
pub use meter::AudioMeter;
pub use mixer::{AudioMixer, MixerInput};
pub use output::AudioOutput;
pub use resampler::{resample_buffer, Resampler};
pub use sync::AudioClock;
pub use waveform::WaveformData;
