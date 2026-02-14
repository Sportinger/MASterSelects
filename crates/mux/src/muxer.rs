//! High-level MP4 muxer API.
//!
//! Usage:
//! ```ignore
//! let mut muxer = Mp4Muxer::new(MuxerConfig { output_path: "output.mp4".into() })?;
//! let vid_track = muxer.add_video_track(video_config)?;
//! let aud_track = muxer.add_audio_track(audio_config)?;
//!
//! // Write samples progressively
//! muxer.write_video_sample(vid_track, &encoded_packet)?;
//! muxer.write_audio_sample(aud_track, &audio_data, pts, duration)?;
//!
//! // Finalize: writes moov box and closes file
//! muxer.finalize()?;
//! ```

use ms_common::{AudioCodec, EncodedPacket, Resolution, Rational, TimeCode, VideoCodec};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::PathBuf;

use crate::atoms::{self, seconds_to_timescale, VIDEO_TIMESCALE};
use crate::error::{MuxError, MuxResult};
use crate::mp4::{self, SampleInfo, TrackHandler, TrackInfo};

/// Configuration for creating an MP4 muxer.
#[derive(Clone, Debug)]
pub struct MuxerConfig {
    /// Output file path.
    pub output_path: PathBuf,
}

/// Video track configuration.
#[derive(Clone, Debug)]
pub struct VideoTrackConfig {
    /// Video codec (H264, H265).
    pub codec: VideoCodec,
    /// Video resolution.
    pub resolution: Resolution,
    /// Frame rate.
    pub fps: Rational,
    /// H.264 SPS NAL unit (without start code prefix).
    pub sps: Vec<u8>,
    /// H.264 PPS NAL unit (without start code prefix).
    pub pps: Vec<u8>,
}

/// Audio track configuration.
#[derive(Clone, Debug)]
pub struct AudioTrackConfig {
    /// Audio codec (Aac, Opus).
    pub codec: AudioCodec,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Codec-specific config data (e.g. AudioSpecificConfig for AAC).
    pub config_data: Vec<u8>,
}

/// Internal track state during muxing.
struct TrackState {
    /// Track ID (1-based).
    track_id: u32,
    /// Track timescale.
    timescale: u32,
    /// Accumulated duration in timescale units.
    total_duration: u64,
    /// Handler info (video or audio config).
    handler: TrackHandler,
    /// Collected sample metadata.
    samples: Vec<SampleInfo>,
}

/// High-level MP4 muxer that writes to a file.
///
/// Writes data progressively to an mdat box, then writes the moov box
/// at the end during `finalize()`.
pub struct Mp4Muxer {
    /// Buffered writer for the output file.
    writer: BufWriter<File>,
    /// All tracks being written.
    tracks: Vec<TrackState>,
    /// Byte offset where the mdat content starts (after the mdat header).
    mdat_data_start: u64,
    /// Whether we are using a 64-bit mdat header (for files > ~4GB).
    #[allow(dead_code)]
    mdat_large: bool,
    /// Position of the mdat size field to patch at finalize.
    mdat_size_pos: u64,
    /// Next track ID to assign.
    next_track_id: u32,
    /// Whether finalize has been called.
    finalized: bool,
}

impl Mp4Muxer {
    /// Create a new MP4 muxer writing to the given file.
    pub fn new(config: MuxerConfig) -> MuxResult<Self> {
        let file = File::create(&config.output_path).map_err(|e| {
            MuxError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to create output file {:?}: {}", config.output_path, e),
            ))
        })?;
        let mut writer = BufWriter::new(file);

        // Write ftyp box
        mp4::write_ftyp(&mut writer)?;

        // Start mdat box with a 64-bit header placeholder.
        // We use a large box header because we don't know the final size yet,
        // and it might exceed 4GB. We'll patch the size in finalize().
        let mdat_size_pos = atoms::large_box_size_placeholder(&mut writer, b"mdat")?;
        let mdat_data_start = writer.stream_position()?;

        Ok(Self {
            writer,
            tracks: Vec::new(),
            mdat_data_start,
            mdat_large: true,
            mdat_size_pos,
            next_track_id: 1,
            finalized: false,
        })
    }

    /// Add a video track. Returns the track ID for writing samples.
    pub fn add_video_track(&mut self, config: VideoTrackConfig) -> MuxResult<u32> {
        if self.finalized {
            return Err(MuxError::InvalidConfig(
                "Cannot add track after finalize".into(),
            ));
        }

        match config.codec {
            VideoCodec::H264 | VideoCodec::H265 => {}
            other => {
                return Err(MuxError::InvalidConfig(format!(
                    "Unsupported video codec for MP4: {:?}",
                    other
                )));
            }
        }

        let track_id = self.next_track_id;
        self.next_track_id += 1;

        let timescale = VIDEO_TIMESCALE;

        self.tracks.push(TrackState {
            track_id,
            timescale,
            total_duration: 0,
            handler: TrackHandler::Video {
                codec: config.codec,
                width: config.resolution.width,
                height: config.resolution.height,
                sps: config.sps,
                pps: config.pps,
            },
            samples: Vec::new(),
        });

        tracing::info!(track_id, codec = ?config.codec, "Added video track");
        Ok(track_id)
    }

    /// Add an audio track. Returns the track ID for writing samples.
    pub fn add_audio_track(&mut self, config: AudioTrackConfig) -> MuxResult<u32> {
        if self.finalized {
            return Err(MuxError::InvalidConfig(
                "Cannot add track after finalize".into(),
            ));
        }

        match config.codec {
            AudioCodec::Aac | AudioCodec::Opus => {}
            other => {
                return Err(MuxError::InvalidConfig(format!(
                    "Unsupported audio codec for MP4: {:?}",
                    other
                )));
            }
        }

        let track_id = self.next_track_id;
        self.next_track_id += 1;

        let timescale = config.sample_rate;

        self.tracks.push(TrackState {
            track_id,
            timescale,
            total_duration: 0,
            handler: TrackHandler::Audio {
                codec: config.codec,
                sample_rate: config.sample_rate,
                channels: config.channels,
                config_data: config.config_data,
            },
            samples: Vec::new(),
        });

        tracing::info!(track_id, codec = ?config.codec, "Added audio track");
        Ok(track_id)
    }

    /// Write a video sample (encoded frame) to the mdat box.
    pub fn write_video_sample(
        &mut self,
        track_id: u32,
        packet: &EncodedPacket,
    ) -> MuxResult<()> {
        if self.finalized {
            return Err(MuxError::InvalidConfig(
                "Cannot write sample after finalize".into(),
            ));
        }

        let track = self
            .find_track_mut(track_id)?;
        let timescale = track.timescale;

        // Calculate duration from PTS difference with previous sample,
        // or use a default based on the video timescale.
        let duration = if let Some(last_sample) = track.samples.last() {
            let last_pts_ticks = track.total_duration.saturating_sub(last_sample.duration as u64);
            let this_pts_ticks = seconds_to_timescale(packet.pts.as_secs(), timescale);
            let dur = this_pts_ticks.saturating_sub(last_pts_ticks);
            if dur == 0 { last_sample.duration } else { dur as u32 }
        } else {
            // First sample: use a default duration (e.g. 1 frame at 30fps = 3000 ticks at 90kHz)
            3000
        };

        // Composition offset: PTS - DTS in timescale units
        let pts_ticks = seconds_to_timescale(packet.pts.as_secs(), timescale);
        let dts_ticks = seconds_to_timescale(packet.dts.as_secs(), timescale);
        let composition_offset = pts_ticks as i64 - dts_ticks as i64;

        // Write data to mdat
        let offset = self.writer.stream_position()?;
        self.writer.write_all(&packet.data)?;

        let track = self.find_track_mut(track_id)?;
        track.samples.push(SampleInfo {
            offset,
            size: packet.data.len() as u32,
            duration,
            composition_offset: composition_offset as i32,
            is_sync: packet.is_keyframe,
        });
        track.total_duration += duration as u64;

        Ok(())
    }

    /// Write an audio sample to the mdat box.
    pub fn write_audio_sample(
        &mut self,
        track_id: u32,
        data: &[u8],
        pts: TimeCode,
        duration: f64,
    ) -> MuxResult<()> {
        if self.finalized {
            return Err(MuxError::InvalidConfig(
                "Cannot write sample after finalize".into(),
            ));
        }

        let track = self.find_track_mut(track_id)?;
        let timescale = track.timescale;
        let duration_ticks = seconds_to_timescale(duration, timescale) as u32;

        // Write data to mdat
        let offset = self.writer.stream_position()?;
        self.writer.write_all(data)?;

        let track = self.find_track_mut(track_id)?;
        let _pts_ticks = seconds_to_timescale(pts.as_secs(), timescale);

        track.samples.push(SampleInfo {
            offset,
            size: data.len() as u32,
            duration: duration_ticks,
            composition_offset: 0, // Audio typically has no composition offset
            is_sync: true,         // Audio samples are always sync
        });
        track.total_duration += duration_ticks as u64;

        Ok(())
    }

    /// Finalize the MP4 file: patch mdat size, write moov box, flush and close.
    pub fn finalize(mut self) -> MuxResult<()> {
        if self.finalized {
            return Err(MuxError::InvalidConfig("Already finalized".into()));
        }

        // Patch the mdat box size
        atoms::fill_large_box_size(&mut self.writer, self.mdat_size_pos)?;

        // Build TrackInfo structs for the moov writer
        let track_infos: Vec<TrackInfo> = self
            .tracks
            .iter()
            .map(|t| TrackInfo {
                track_id: t.track_id,
                timescale: t.timescale,
                duration: t.total_duration,
                handler: t.handler.clone(),
                samples: t.samples.clone(),
            })
            .collect();

        // Write moov box at end of file
        mp4::write_moov(&mut self.writer, &track_infos)?;

        // Flush everything
        self.writer.flush()?;

        tracing::info!(
            tracks = track_infos.len(),
            "MP4 file finalized successfully"
        );

        // Mark as finalized (prevents double-finalize via Drop if we had one)
        self.finalized = true;

        Ok(())
    }

    /// Get the number of samples written to a track.
    pub fn track_sample_count(&self, track_id: u32) -> MuxResult<usize> {
        let track = self.find_track(track_id)?;
        Ok(track.samples.len())
    }

    /// Get the total bytes written to mdat so far.
    pub fn mdat_bytes_written(&mut self) -> MuxResult<u64> {
        let current = self.writer.stream_position()?;
        Ok(current - self.mdat_data_start)
    }

    /// Find a track by ID (immutable).
    fn find_track(&self, track_id: u32) -> MuxResult<&TrackState> {
        self.tracks
            .iter()
            .find(|t| t.track_id == track_id)
            .ok_or_else(|| MuxError::TrackError(format!("Track {} not found", track_id)))
    }

    /// Find a track by ID (mutable).
    fn find_track_mut(&mut self, track_id: u32) -> MuxResult<&mut TrackState> {
        self.tracks
            .iter_mut()
            .find(|t| t.track_id == track_id)
            .ok_or_else(|| MuxError::TrackError(format!("Track {} not found", track_id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Helper: create a temporary file path for testing.
    fn temp_mp4_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ms_mux_test_{}.mp4", name));
        path
    }

    /// Helper: create a minimal SPS for H.264 (Baseline profile, level 3.1).
    fn test_sps() -> Vec<u8> {
        vec![0x67, 0x42, 0xC0, 0x1F, 0xDA, 0x02, 0x80, 0xF6, 0xC0, 0x44, 0x00, 0x00]
    }

    /// Helper: create a minimal PPS for H.264.
    fn test_pps() -> Vec<u8> {
        vec![0x68, 0xCE, 0x38, 0x80]
    }

    /// Helper: create a minimal AAC AudioSpecificConfig (LC, 44100Hz, stereo).
    fn test_aac_config() -> Vec<u8> {
        vec![0x12, 0x10] // AAC-LC, 44100, 2ch
    }

    /// Helper: create a fake encoded video packet.
    fn fake_video_packet(pts_secs: f64, is_keyframe: bool) -> EncodedPacket {
        EncodedPacket {
            data: vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, 0xCC], // fake NAL
            pts: TimeCode::from_secs(pts_secs),
            dts: TimeCode::from_secs(pts_secs),
            is_keyframe,
        }
    }

    #[test]
    fn test_create_muxer() {
        let path = temp_mp4_path("create");
        let muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();
        muxer.finalize().unwrap();
        assert!(path.exists());
        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_add_video_track() {
        let path = temp_mp4_path("add_video");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let track_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        assert_eq!(track_id, 1);
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_add_audio_track() {
        let path = temp_mp4_path("add_audio");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let track_id = muxer
            .add_audio_track(AudioTrackConfig {
                codec: AudioCodec::Aac,
                sample_rate: 44100,
                channels: 2,
                config_data: test_aac_config(),
            })
            .unwrap();

        assert_eq!(track_id, 1);
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_add_multiple_tracks() {
        let path = temp_mp4_path("multi_track");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let vid_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();
        assert_eq!(vid_id, 1);

        let aud_id = muxer
            .add_audio_track(AudioTrackConfig {
                codec: AudioCodec::Aac,
                sample_rate: 44100,
                channels: 2,
                config_data: test_aac_config(),
            })
            .unwrap();
        assert_eq!(aud_id, 2);

        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unsupported_video_codec() {
        let path = temp_mp4_path("unsupported_video");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let result = muxer.add_video_track(VideoTrackConfig {
            codec: VideoCodec::Vp9,
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            sps: vec![],
            pps: vec![],
        });
        assert!(result.is_err());
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unsupported_audio_codec() {
        let path = temp_mp4_path("unsupported_audio");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let result = muxer.add_audio_track(AudioTrackConfig {
            codec: AudioCodec::Mp3,
            sample_rate: 44100,
            channels: 2,
            config_data: vec![],
        });
        assert!(result.is_err());
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_video_samples() {
        let path = temp_mp4_path("video_samples");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let track_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        // Write 10 frames
        for i in 0..10 {
            let packet = fake_video_packet(i as f64 / 30.0, i % 5 == 0);
            muxer.write_video_sample(track_id, &packet).unwrap();
        }

        assert_eq!(muxer.track_sample_count(track_id).unwrap(), 10);
        muxer.finalize().unwrap();

        // Verify file exists and has reasonable size
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 100); // Should have more than just headers
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_audio_samples() {
        let path = temp_mp4_path("audio_samples");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let track_id = muxer
            .add_audio_track(AudioTrackConfig {
                codec: AudioCodec::Aac,
                sample_rate: 44100,
                channels: 2,
                config_data: test_aac_config(),
            })
            .unwrap();

        // Write 20 audio frames (AAC uses 1024 samples per frame)
        let frame_duration = 1024.0 / 44100.0;
        for i in 0..20 {
            let fake_data = vec![0xAA; 512]; // fake AAC frame
            let pts = TimeCode::from_secs(i as f64 * frame_duration);
            muxer
                .write_audio_sample(track_id, &fake_data, pts, frame_duration)
                .unwrap();
        }

        assert_eq!(muxer.track_sample_count(track_id).unwrap(), 20);
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_write_to_invalid_track() {
        let path = temp_mp4_path("invalid_track");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let packet = fake_video_packet(0.0, true);
        let result = muxer.write_video_sample(999, &packet);
        assert!(result.is_err());
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_track_sample_count_invalid_track() {
        let path = temp_mp4_path("count_invalid");
        let muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let result = muxer.track_sample_count(42);
        assert!(result.is_err());
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mdat_bytes_written() {
        let path = temp_mp4_path("mdat_bytes");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let track_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        assert_eq!(muxer.mdat_bytes_written().unwrap(), 0);

        let packet = fake_video_packet(0.0, true);
        let data_len = packet.data.len() as u64;
        muxer.write_video_sample(track_id, &packet).unwrap();

        assert_eq!(muxer.mdat_bytes_written().unwrap(), data_len);
        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_complete_mux_roundtrip() {
        let path = temp_mp4_path("roundtrip");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        // Add video track
        let vid_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::new(640, 480),
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        // Add audio track
        let aud_id = muxer
            .add_audio_track(AudioTrackConfig {
                codec: AudioCodec::Aac,
                sample_rate: 44100,
                channels: 2,
                config_data: test_aac_config(),
            })
            .unwrap();

        // Write interleaved samples
        for i in 0..30 {
            let packet = fake_video_packet(i as f64 / 30.0, i % 10 == 0);
            muxer.write_video_sample(vid_id, &packet).unwrap();

            if i % 3 == 0 {
                let audio_data = vec![0xBB; 256];
                let pts = TimeCode::from_secs(i as f64 / 30.0);
                muxer
                    .write_audio_sample(aud_id, &audio_data, pts, 1024.0 / 44100.0)
                    .unwrap();
            }
        }

        muxer.finalize().unwrap();

        // Read file and verify box structure
        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        // Verify ftyp is first box
        assert_eq!(&file_data[4..8], b"ftyp");

        // Verify mdat exists
        assert!(file_data.windows(4).any(|w| w == b"mdat"));

        // Verify moov exists at end
        assert!(file_data.windows(4).any(|w| w == b"moov"));

        // Verify moov contains expected sub-boxes
        assert!(file_data.windows(4).any(|w| w == b"mvhd"));
        assert!(file_data.windows(4).any(|w| w == b"trak"));
        assert!(file_data.windows(4).any(|w| w == b"tkhd"));
        assert!(file_data.windows(4).any(|w| w == b"mdia"));
        assert!(file_data.windows(4).any(|w| w == b"mdhd"));
        assert!(file_data.windows(4).any(|w| w == b"hdlr"));
        assert!(file_data.windows(4).any(|w| w == b"minf"));
        assert!(file_data.windows(4).any(|w| w == b"stbl"));
        assert!(file_data.windows(4).any(|w| w == b"stsd"));
        assert!(file_data.windows(4).any(|w| w == b"stts"));
        assert!(file_data.windows(4).any(|w| w == b"stsc"));
        assert!(file_data.windows(4).any(|w| w == b"stsz"));
        assert!(file_data.windows(4).any(|w| w == b"avcC"));
        assert!(file_data.windows(4).any(|w| w == b"esds"));

        // Verify video handler "vide" and audio handler "soun"
        assert!(file_data.windows(4).any(|w| w == b"vide"));
        assert!(file_data.windows(4).any(|w| w == b"soun"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_video_only_mux() {
        let path = temp_mp4_path("video_only");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let vid_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_24,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        for i in 0..5 {
            let packet = fake_video_packet(i as f64 / 24.0, i == 0);
            muxer.write_video_sample(vid_id, &packet).unwrap();
        }

        muxer.finalize().unwrap();

        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        // Should have ftyp, mdat, moov
        assert_eq!(&file_data[4..8], b"ftyp");
        assert!(file_data.windows(4).any(|w| w == b"mdat"));
        assert!(file_data.windows(4).any(|w| w == b"moov"));

        // Should NOT have audio handler
        assert!(!file_data.windows(4).any(|w| w == b"soun"));
        // Should have video handler
        assert!(file_data.windows(4).any(|w| w == b"vide"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_audio_only_mux() {
        let path = temp_mp4_path("audio_only");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let aud_id = muxer
            .add_audio_track(AudioTrackConfig {
                codec: AudioCodec::Aac,
                sample_rate: 48000,
                channels: 2,
                config_data: test_aac_config(),
            })
            .unwrap();

        let frame_duration = 1024.0 / 48000.0;
        for i in 0..10 {
            let data = vec![0xCC; 128];
            let pts = TimeCode::from_secs(i as f64 * frame_duration);
            muxer
                .write_audio_sample(aud_id, &data, pts, frame_duration)
                .unwrap();
        }

        muxer.finalize().unwrap();

        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        assert!(file_data.windows(4).any(|w| w == b"soun"));
        assert!(!file_data.windows(4).any(|w| w == b"vide"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_empty_tracks_finalize() {
        let path = temp_mp4_path("empty_tracks");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        // Add tracks but don't write any samples
        muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        muxer.finalize().unwrap();
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_single_frame_video() {
        let path = temp_mp4_path("single_frame");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let vid_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::new(320, 240),
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        let packet = fake_video_packet(0.0, true);
        muxer.write_video_sample(vid_id, &packet).unwrap();
        assert_eq!(muxer.track_sample_count(vid_id).unwrap(), 1);

        muxer.finalize().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_h265_video_track() {
        let path = temp_mp4_path("h265");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let vid_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H265,
                resolution: Resolution::UHD,
                fps: Rational::FPS_60,
                sps: vec![0x42, 0x01, 0x01, 0x01],
                pps: vec![0x44, 0x01, 0xC1],
            })
            .unwrap();

        for i in 0..5 {
            let packet = fake_video_packet(i as f64 / 60.0, i == 0);
            muxer.write_video_sample(vid_id, &packet).unwrap();
        }

        muxer.finalize().unwrap();

        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        assert!(file_data.windows(4).any(|w| w == b"hvc1"));
        assert!(file_data.windows(4).any(|w| w == b"hvcC"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_opus_audio_track() {
        let path = temp_mp4_path("opus");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let aud_id = muxer
            .add_audio_track(AudioTrackConfig {
                codec: AudioCodec::Opus,
                sample_rate: 48000,
                channels: 2,
                config_data: vec![],
            })
            .unwrap();

        let data = vec![0xDD; 64];
        muxer
            .write_audio_sample(aud_id, &data, TimeCode::ZERO, 0.02)
            .unwrap();

        muxer.finalize().unwrap();

        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        assert!(file_data.windows(4).any(|w| w == b"Opus"));
        assert!(file_data.windows(4).any(|w| w == b"dOps"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ftyp_is_first_box_in_file() {
        let path = temp_mp4_path("ftyp_first");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        muxer.finalize().unwrap();

        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        // ftyp must be the very first box
        assert_eq!(&file_data[4..8], b"ftyp");
        let ftyp_size = u32::from_be_bytes(file_data[0..4].try_into().unwrap());
        assert_eq!(ftyp_size, 28);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_moov_is_last_box_in_file() {
        let path = temp_mp4_path("moov_last");
        let mut muxer = Mp4Muxer::new(MuxerConfig {
            output_path: path.clone(),
        })
        .unwrap();

        let vid_id = muxer
            .add_video_track(VideoTrackConfig {
                codec: VideoCodec::H264,
                resolution: Resolution::HD,
                fps: Rational::FPS_30,
                sps: test_sps(),
                pps: test_pps(),
            })
            .unwrap();

        let packet = fake_video_packet(0.0, true);
        muxer.write_video_sample(vid_id, &packet).unwrap();
        muxer.finalize().unwrap();

        let mut file_data = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut file_data)
            .unwrap();

        // Walk the top-level boxes to find the last one
        let mut offset = 0;
        let mut last_box_type = [0u8; 4];
        while offset + 8 <= file_data.len() {
            let size_bytes: [u8; 4] = file_data[offset..offset + 4].try_into().unwrap();
            let size = u32::from_be_bytes(size_bytes);
            last_box_type.copy_from_slice(&file_data[offset + 4..offset + 8]);

            if size == 1 {
                // Extended size
                if offset + 16 > file_data.len() {
                    break;
                }
                let ext_size = u64::from_be_bytes(
                    file_data[offset + 8..offset + 16].try_into().unwrap(),
                );
                offset += ext_size as usize;
            } else if size == 0 {
                // Box extends to EOF
                break;
            } else {
                offset += size as usize;
            }
        }

        assert_eq!(&last_box_type, b"moov");
        std::fs::remove_file(&path).ok();
    }
}
