//! `ms-mux` — MP4 container muxer for the MasterSelects native engine.
//!
//! This crate handles combining encoded video and audio streams into a playable
//! MP4 file (ISO Base Media File Format / ISO 14496-12).
//!
//! # Architecture
//!
//! - **No FFmpeg dependency** — pure Rust MP4 box writing
//! - **Progressive write** — mdat box data is written as samples arrive
//! - **Moov-at-end** — the moov (metadata) box is written during `finalize()`
//! - **Dual codec support** — H.264 (avcC), H.265 (hvcC) for video; AAC (esds), Opus (dOps) for audio
//!
//! # Usage
//!
//! ```ignore
//! use ms_mux::{Mp4Muxer, MuxerConfig, VideoTrackConfig, AudioTrackConfig};
//! use ms_common::{VideoCodec, AudioCodec, Resolution, Rational};
//!
//! let mut muxer = Mp4Muxer::new(MuxerConfig {
//!     output_path: "output.mp4".into(),
//! })?;
//!
//! let vid_track = muxer.add_video_track(VideoTrackConfig {
//!     codec: VideoCodec::H264,
//!     resolution: Resolution::HD,
//!     fps: Rational::FPS_30,
//!     sps: sps_data,
//!     pps: pps_data,
//! })?;
//!
//! // Write encoded packets progressively
//! muxer.write_video_sample(vid_track, &encoded_packet)?;
//!
//! // Finalize writes moov and closes file
//! muxer.finalize()?;
//! ```

pub mod atoms;
pub mod error;
pub mod mp4;
pub mod muxer;

// Re-export primary API types
pub use error::MuxError;
pub use muxer::{AudioTrackConfig, Mp4Muxer, MuxerConfig, VideoTrackConfig};
