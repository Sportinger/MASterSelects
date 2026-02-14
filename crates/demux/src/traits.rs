//! Demuxer trait definition.

use ms_common::{AudioPacket, ContainerInfo, VideoPacket};

/// Trait for container demuxers (MP4, MKV).
pub trait Demuxer {
    /// Probe the container and return stream information.
    fn probe(&self) -> &ContainerInfo;

    /// Read the next video packet (NAL units in Annex-B format).
    fn next_video_packet(&mut self) -> Option<VideoPacket>;

    /// Read the next audio packet (compressed audio data).
    fn next_audio_packet(&mut self) -> Option<AudioPacket>;

    /// Seek to the nearest keyframe before the given timestamp.
    /// This seeks both video and audio streams.
    fn seek(&mut self, time_secs: f64) -> Result<(), ms_common::DemuxError>;

    /// Returns true if this container has an audio track.
    fn has_audio(&self) -> bool;

    /// Reset the demuxer to the beginning (both video and audio).
    fn reset(&mut self);
}
