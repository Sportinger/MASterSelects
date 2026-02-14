//! MP4/MOV (ISO BMFF) demuxer.
//!
//! Parses MP4 containers and yields video samples as `VideoPacket`s
//! with NAL data converted from AVCC to Annex-B format, ready for
//! hardware decoders (NVDEC, Vulkan Video).

pub mod boxes;
pub mod sample;

use boxes::{
    parse_ftyp, parse_moov, read_box_header, skip_box, AVC1, AVC3, FTYP, HEV1, HVC1, MOOV, MP4A,
    OPUS, ParsedAudioTrack, ParsedMoov, ParsedVideoTrack,
};
use ms_common::{
    AudioCodec, AudioPacket, AudioStreamInfo, ContainerInfo, DemuxError, PixelFormat, Rational,
    Resolution, TimeCode, VideoCodec, VideoPacket, VideoStreamInfo,
};
use sample::SampleTable;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::nal;
use crate::traits::Demuxer;

/// MP4 demuxer — reads an MP4 file and yields video and audio packets.
pub struct Mp4Demuxer {
    /// Path to the MP4 file (kept for diagnostics/logging).
    #[allow(dead_code)]
    path: PathBuf,
    /// Buffered file reader.
    reader: BufReader<File>,
    /// Parsed moov box data.
    moov: ParsedMoov,
    /// Pre-computed sample table for the primary video track.
    sample_table: SampleTable,
    /// Index of the video track we're reading (in moov.video_tracks).
    video_track_idx: usize,
    /// Current video sample index (next sample to read).
    current_sample: usize,
    /// Pre-computed sample table for the primary audio track (if present).
    audio_sample_table: Option<SampleTable>,
    /// Index of the audio track we're reading (in moov.audio_tracks).
    audio_track_idx: Option<usize>,
    /// Current audio sample index (next sample to read).
    current_audio_sample: usize,
    /// Container info (for the Demuxer trait).
    container_info: ContainerInfo,
}

impl Mp4Demuxer {
    /// Open an MP4 file and parse its structure.
    pub fn open(path: &Path) -> Result<Self, DemuxError> {
        info!("Opening MP4 file: {}", path.display());

        let file = File::open(path).map_err(DemuxError::Io)?;
        let mut reader = BufReader::new(file);

        // Step 1: Scan top-level boxes to find moov
        let moov = find_and_parse_moov(&mut reader)?;

        // Step 2: Find the first video track
        if moov.video_tracks.is_empty() {
            return Err(DemuxError::NoVideoTrack);
        }
        let video_track_idx = 0;
        let track = &moov.video_tracks[video_track_idx];

        // Step 3: Build video sample table
        let sample_table = SampleTable::build(track)?;

        info!(
            "MP4: {} video track(s), primary track: {}x{}, {} samples, {:.2}s",
            moov.video_tracks.len(),
            track.width,
            track.height,
            sample_table.samples.len(),
            sample_table.duration_secs()
        );

        // Step 4: Build audio sample table (if audio track exists)
        let (audio_track_idx, audio_sample_table) = if !moov.audio_tracks.is_empty() {
            let audio_idx = 0;
            let audio_track = &moov.audio_tracks[audio_idx];
            let audio_st = SampleTable::build_from_audio(audio_track)?;

            info!(
                "MP4: {} audio track(s), primary: {} channels, {}Hz, {} samples, {:.2}s",
                moov.audio_tracks.len(),
                audio_track.sample_desc.channel_count,
                audio_track.sample_desc.sample_rate,
                audio_st.samples.len(),
                audio_st.duration_secs()
            );

            (Some(audio_idx), Some(audio_st))
        } else {
            (None, None)
        };

        // Step 5: Build container info
        let container_info = build_container_info(&moov);

        Ok(Mp4Demuxer {
            path: path.to_path_buf(),
            reader,
            moov,
            sample_table,
            video_track_idx,
            current_sample: 0,
            audio_sample_table,
            audio_track_idx,
            current_audio_sample: 0,
            container_info,
        })
    }

    /// Get a reference to the active video track.
    fn video_track(&self) -> &ParsedVideoTrack {
        &self.moov.video_tracks[self.video_track_idx]
    }

    /// Determine the VideoCodec from the sample description FourCC.
    fn video_codec(&self) -> VideoCodec {
        codec_from_fourcc(self.video_track().sample_desc.codec_fourcc)
    }

    /// Read a specific sample's raw data from the file.
    fn read_sample_data(&mut self, sample_idx: usize) -> Result<Vec<u8>, DemuxError> {
        let entry = &self.sample_table.samples[sample_idx];
        self.reader
            .seek(SeekFrom::Start(entry.offset))
            .map_err(DemuxError::Io)?;

        let mut data = vec![0u8; entry.size as usize];
        self.reader.read_exact(&mut data).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                DemuxError::TruncatedData {
                    expected: entry.size as usize,
                    got: 0, // approximate
                }
            } else {
                DemuxError::Io(e)
            }
        })?;

        Ok(data)
    }

    /// Get a reference to the active audio track, if present.
    #[allow(dead_code)]
    fn audio_track(&self) -> Option<&ParsedAudioTrack> {
        self.audio_track_idx
            .map(|idx| &self.moov.audio_tracks[idx])
    }

    /// Determine the AudioCodec from the sample description FourCC.
    #[allow(dead_code)]
    fn audio_codec(&self) -> Option<AudioCodec> {
        self.audio_track()
            .map(|t| audio_codec_from_fourcc(t.sample_desc.codec_fourcc))
    }

    /// Read a specific audio sample's raw data from the file.
    fn read_audio_sample_data(&mut self, sample_idx: usize) -> Result<Vec<u8>, DemuxError> {
        let audio_st = self
            .audio_sample_table
            .as_ref()
            .ok_or(DemuxError::NoAudioTrack)?;
        let entry = &audio_st.samples[sample_idx];

        self.reader
            .seek(SeekFrom::Start(entry.offset))
            .map_err(DemuxError::Io)?;

        let mut data = vec![0u8; entry.size as usize];
        self.reader.read_exact(&mut data).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                DemuxError::TruncatedData {
                    expected: entry.size as usize,
                    got: 0,
                }
            } else {
                DemuxError::Io(e)
            }
        })?;

        Ok(data)
    }

    /// Convert raw sample data (AVCC format) to Annex-B and optionally prepend SPS/PPS.
    fn convert_sample_to_annexb(
        &self,
        raw_data: &[u8],
        is_keyframe: bool,
    ) -> Result<Vec<u8>, DemuxError> {
        let track = self.video_track();

        // Get length_size from avcC config
        let length_size = track
            .sample_desc
            .avcc
            .as_ref()
            .map(|c| c.length_size())
            .unwrap_or(4);

        // Convert AVCC NAL units to Annex-B
        let annexb_data = nal::avcc_to_annexb(raw_data, length_size);

        // Prepend SPS/PPS on keyframes
        if is_keyframe {
            if let Some(avcc) = &track.sample_desc.avcc {
                if let (Some(sps), Some(pps)) =
                    (avcc.sps_list.first(), avcc.pps_list.first())
                {
                    return Ok(nal::prepend_sps_pps(sps, pps, &annexb_data));
                }
            }
        }

        Ok(annexb_data)
    }
}

impl Demuxer for Mp4Demuxer {
    fn probe(&self) -> &ContainerInfo {
        &self.container_info
    }

    fn next_video_packet(&mut self) -> Option<VideoPacket> {
        if self.current_sample >= self.sample_table.samples.len() {
            return None;
        }

        let sample_idx = self.current_sample;
        let entry = &self.sample_table.samples[sample_idx];
        let is_sync = entry.is_sync;
        let pts_secs = self.sample_table.ticks_to_secs(entry.cts);
        let dts_secs = self.sample_table.ticks_to_secs(entry.dts as i64);

        // Read raw sample data
        let raw_data = match self.read_sample_data(sample_idx) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("Failed to read sample {}: {}", sample_idx, e);
                return None;
            }
        };

        // Convert to Annex-B
        let annexb_data = match self.convert_sample_to_annexb(&raw_data, is_sync) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("Failed to convert sample {} to Annex-B: {}", sample_idx, e);
                return None;
            }
        };

        self.current_sample += 1;

        Some(VideoPacket {
            data: annexb_data,
            pts: TimeCode(pts_secs),
            dts: TimeCode(dts_secs),
            is_keyframe: is_sync,
            codec: self.video_codec(),
        })
    }

    fn next_audio_packet(&mut self) -> Option<AudioPacket> {
        let audio_st = self.audio_sample_table.as_ref()?;
        let audio_track_idx = self.audio_track_idx?;
        let audio_track = &self.moov.audio_tracks[audio_track_idx];

        if self.current_audio_sample >= audio_st.samples.len() {
            return None;
        }

        let sample_idx = self.current_audio_sample;
        let entry = &audio_st.samples[sample_idx];
        let pts_secs = audio_st.ticks_to_secs(entry.cts);

        // Extract metadata before mutable borrow
        let codec = audio_codec_from_fourcc(audio_track.sample_desc.codec_fourcc);
        let sample_rate = audio_track.sample_desc.sample_rate;
        let channels = audio_track.sample_desc.channel_count;

        // Read raw audio sample data (requires &mut self for reader)
        let raw_data = match self.read_audio_sample_data(sample_idx) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("Failed to read audio sample {}: {}", sample_idx, e);
                return None;
            }
        };

        self.current_audio_sample += 1;

        Some(AudioPacket {
            data: raw_data,
            pts: TimeCode(pts_secs),
            codec,
            sample_rate,
            channels,
        })
    }

    fn seek(&mut self, time_secs: f64) -> Result<(), DemuxError> {
        // Seek video
        match self.sample_table.find_sync_at_or_before(time_secs) {
            Some(idx) => {
                debug!(
                    "Seeking video to sample {} (sync at or before {:.3}s)",
                    idx, time_secs
                );
                self.current_sample = idx;
            }
            None => {
                self.current_sample = 0;
            }
        }

        // Seek audio (if present)
        if let Some(audio_st) = &self.audio_sample_table {
            // Audio samples are typically all sync points, so find the sample
            // closest to the requested time.
            match audio_st.find_sample_at_time(time_secs) {
                Some(idx) => {
                    debug!(
                        "Seeking audio to sample {} (at or before {:.3}s)",
                        idx, time_secs
                    );
                    self.current_audio_sample = idx;
                }
                None => {
                    self.current_audio_sample = 0;
                }
            }
        }

        Ok(())
    }

    fn has_audio(&self) -> bool {
        self.audio_track_idx.is_some()
    }

    fn reset(&mut self) {
        self.current_sample = 0;
        self.current_audio_sample = 0;
    }
}

// ─── Helper functions ───────────────────────────────────────────────

/// Scan top-level boxes and parse the moov box.
fn find_and_parse_moov<R: Read + Seek>(reader: &mut R) -> Result<ParsedMoov, DemuxError> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(DemuxError::Io)?;

    loop {
        let header = match read_box_header(reader)? {
            Some(h) => h,
            None => {
                return Err(DemuxError::InvalidStructure {
                    offset: 0,
                    reason: "No moov box found in file".into(),
                });
            }
        };

        match header.box_type {
            FTYP => {
                let _ftyp = parse_ftyp(reader, &header)?;
                // Seek to end of ftyp box
                if let Some(end) = header.end_offset() {
                    reader
                        .seek(SeekFrom::Start(end))
                        .map_err(DemuxError::Io)?;
                }
            }
            MOOV => {
                return parse_moov(reader, &header);
            }
            _ => {
                skip_box(reader, &header)?;
            }
        }
    }
}

/// Map a codec FourCC to our VideoCodec enum.
fn codec_from_fourcc(fourcc: u32) -> VideoCodec {
    match fourcc {
        AVC1 | AVC3 => VideoCodec::H264,
        HEV1 | HVC1 => VideoCodec::H265,
        _ => VideoCodec::H264, // fallback
    }
}

/// Map an audio codec FourCC to our AudioCodec enum.
fn audio_codec_from_fourcc(fourcc: u32) -> AudioCodec {
    match fourcc {
        MP4A => AudioCodec::Aac,
        OPUS => AudioCodec::Opus,
        _ => AudioCodec::Aac, // fallback for unknown audio codecs
    }
}

/// Build ContainerInfo from parsed moov data.
fn build_container_info(moov: &ParsedMoov) -> ContainerInfo {
    let global_duration_secs = if moov.timescale > 0 {
        moov.duration as f64 / moov.timescale as f64
    } else {
        0.0
    };

    let video_streams: Vec<VideoStreamInfo> = moov
        .video_tracks
        .iter()
        .map(|track| {
            let codec = codec_from_fourcc(track.sample_desc.codec_fourcc);
            let duration_secs = if track.timescale > 0 {
                track.duration as f64 / track.timescale as f64
            } else {
                0.0
            };

            // Estimate FPS from stts
            let fps = estimate_fps(track);

            // Estimate bitrate
            let total_bytes: u64 = if track.stsz.default_sample_size > 0 {
                track.stsz.default_sample_size as u64 * track.stsz.sample_count as u64
            } else {
                track.stsz.sample_sizes.iter().map(|&s| s as u64).sum()
            };
            let bitrate = if duration_secs > 0.0 {
                (total_bytes as f64 * 8.0 / duration_secs) as u64
            } else {
                0
            };

            // Build extra_data from SPS+PPS
            let extra_data = build_extra_data(track);

            VideoStreamInfo {
                codec,
                resolution: Resolution::new(track.width, track.height),
                fps,
                duration: TimeCode(duration_secs),
                bitrate,
                pixel_format: PixelFormat::Nv12, // typical for HW decoded output
                extra_data,
            }
        })
        .collect();

    let audio_streams: Vec<AudioStreamInfo> = moov
        .audio_tracks
        .iter()
        .map(|track| {
            let codec = audio_codec_from_fourcc(track.sample_desc.codec_fourcc);
            let duration_secs = if track.timescale > 0 {
                track.duration as f64 / track.timescale as f64
            } else {
                0.0
            };

            // Estimate bitrate from total sample sizes
            let total_bytes: u64 = if track.stsz.default_sample_size > 0 {
                track.stsz.default_sample_size as u64 * track.stsz.sample_count as u64
            } else {
                track.stsz.sample_sizes.iter().map(|&s| s as u64).sum()
            };
            let bitrate = if duration_secs > 0.0 {
                (total_bytes as f64 * 8.0 / duration_secs) as u64
            } else {
                0
            };

            // Use sample rate from AAC config if available, otherwise from sample entry
            let sample_rate = track
                .sample_desc
                .aac_config
                .as_ref()
                .and_then(|c| if c.sample_rate > 0 { Some(c.sample_rate) } else { None })
                .unwrap_or(track.sample_desc.sample_rate);

            // Use channel count from AAC config if available, otherwise from sample entry
            let channels = track
                .sample_desc
                .aac_config
                .as_ref()
                .and_then(|c| {
                    if c.channel_config > 0 {
                        Some(c.channel_config as u16)
                    } else {
                        None
                    }
                })
                .unwrap_or(track.sample_desc.channel_count);

            AudioStreamInfo {
                codec,
                sample_rate,
                channels,
                duration: TimeCode(duration_secs),
                bitrate,
            }
        })
        .collect();

    ContainerInfo {
        video_streams,
        audio_streams,
        duration: TimeCode(global_duration_secs),
    }
}

/// Estimate FPS from the stts table.
fn estimate_fps(track: &ParsedVideoTrack) -> Rational {
    if track.stts.is_empty() || track.timescale == 0 {
        return Rational::FPS_30; // default fallback
    }

    // Use the first (most common) delta
    let delta = track.stts[0].sample_delta;
    if delta == 0 {
        return Rational::FPS_30;
    }

    // fps = timescale / delta
    // Try to find a clean rational representation
    let gcd = gcd(track.timescale, delta);
    Rational::new(track.timescale / gcd, delta / gcd)
}

/// Greatest common divisor (Euclidean algorithm).
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Build extra_data (SPS + PPS in Annex-B format) for VideoStreamInfo.
fn build_extra_data(track: &ParsedVideoTrack) -> Vec<u8> {
    let mut data = Vec::new();
    if let Some(avcc) = &track.sample_desc.avcc {
        for sps in &avcc.sps_list {
            data.extend_from_slice(&nal::ANNEXB_START_CODE);
            data.extend_from_slice(sps);
        }
        for pps in &avcc.pps_list {
            data.extend_from_slice(&nal::ANNEXB_START_CODE);
            data.extend_from_slice(pps);
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_from_fourcc() {
        assert_eq!(codec_from_fourcc(AVC1), VideoCodec::H264);
        assert_eq!(codec_from_fourcc(AVC3), VideoCodec::H264);
        assert_eq!(codec_from_fourcc(HEV1), VideoCodec::H265);
        assert_eq!(codec_from_fourcc(HVC1), VideoCodec::H265);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(30000, 1001), 1);
        assert_eq!(gcd(48000, 1000), 1000);
        assert_eq!(gcd(24, 8), 8);
    }

    #[test]
    fn test_estimate_fps() {
        use crate::mp4::boxes::*;

        // 30fps: timescale=30000, delta=1000
        let track = ParsedVideoTrack {
            track_id: 1,
            timescale: 30000,
            duration: 300000,
            width: 1920,
            height: 1080,
            sample_desc: VideoSampleDesc {
                codec_fourcc: AVC1,
                width: 1920,
                height: 1080,
                avcc: None,
                hvcc: None,
            },
            stts: vec![SttsEntry {
                sample_count: 300,
                sample_delta: 1000,
            }],
            ctts: vec![],
            stsc: vec![],
            stsz: StszBox {
                default_sample_size: 0,
                sample_sizes: vec![],
                sample_count: 0,
            },
            chunk_offsets: vec![],
            sync_samples: vec![],
        };

        let fps = estimate_fps(&track);
        assert_eq!(fps.num, 30);
        assert_eq!(fps.den, 1);
    }

    #[test]
    fn test_estimate_fps_29_97() {
        use crate::mp4::boxes::*;

        // 29.97fps: timescale=30000, delta=1001
        let track = ParsedVideoTrack {
            track_id: 1,
            timescale: 30000,
            duration: 300300,
            width: 1920,
            height: 1080,
            sample_desc: VideoSampleDesc {
                codec_fourcc: AVC1,
                width: 1920,
                height: 1080,
                avcc: None,
                hvcc: None,
            },
            stts: vec![SttsEntry {
                sample_count: 300,
                sample_delta: 1001,
            }],
            ctts: vec![],
            stsc: vec![],
            stsz: StszBox {
                default_sample_size: 0,
                sample_sizes: vec![],
                sample_count: 0,
            },
            chunk_offsets: vec![],
            sync_samples: vec![],
        };

        let fps = estimate_fps(&track);
        assert_eq!(fps.num, 30000);
        assert_eq!(fps.den, 1001);
    }

    #[test]
    fn test_fourcc_to_string_coverage() {
        use boxes::fourcc_to_string;
        assert_eq!(fourcc_to_string(AVC1), "avc1");
        assert_eq!(fourcc_to_string(HEV1), "hev1");
    }

    #[test]
    fn test_audio_codec_from_fourcc() {
        assert_eq!(audio_codec_from_fourcc(MP4A), AudioCodec::Aac);
        assert_eq!(audio_codec_from_fourcc(OPUS), AudioCodec::Opus);
        // Fallback for unknown
        assert_eq!(audio_codec_from_fourcc(0x12345678), AudioCodec::Aac);
    }

    #[test]
    fn test_build_container_info_with_audio() {
        use crate::mp4::boxes::*;

        let moov = ParsedMoov {
            timescale: 90000,
            duration: 900000, // 10 seconds
            video_tracks: vec![ParsedVideoTrack {
                track_id: 1,
                timescale: 30000,
                duration: 300000,
                width: 1920,
                height: 1080,
                sample_desc: VideoSampleDesc {
                    codec_fourcc: AVC1,
                    width: 1920,
                    height: 1080,
                    avcc: None,
                    hvcc: None,
                },
                stts: vec![SttsEntry {
                    sample_count: 300,
                    sample_delta: 1000,
                }],
                ctts: vec![],
                stsc: vec![],
                stsz: StszBox {
                    default_sample_size: 5000,
                    sample_sizes: vec![],
                    sample_count: 300,
                },
                chunk_offsets: vec![],
                sync_samples: vec![],
            }],
            audio_tracks: vec![ParsedAudioTrack {
                track_id: 2,
                timescale: 44100,
                duration: 441000, // 10 seconds
                sample_desc: AudioSampleDesc {
                    codec_fourcc: MP4A,
                    channel_count: 2,
                    sample_size: 16,
                    sample_rate: 44100,
                    aac_config: Some(AacConfig {
                        audio_object_type: 2,
                        sampling_frequency_index: 4,
                        sample_rate: 44100,
                        channel_config: 2,
                        raw_config: vec![0x12, 0x10],
                    }),
                    opus_config: None,
                },
                stts: vec![SttsEntry {
                    sample_count: 431,
                    sample_delta: 1024,
                }],
                ctts: vec![],
                stsc: vec![],
                stsz: StszBox {
                    default_sample_size: 400,
                    sample_sizes: vec![],
                    sample_count: 431,
                },
                chunk_offsets: vec![],
                sync_samples: vec![],
            }],
        };

        let info = build_container_info(&moov);

        // Check video streams
        assert_eq!(info.video_streams.len(), 1);
        assert_eq!(info.video_streams[0].codec, VideoCodec::H264);
        assert_eq!(info.video_streams[0].resolution.width, 1920);

        // Check audio streams
        assert_eq!(info.audio_streams.len(), 1);
        assert_eq!(info.audio_streams[0].codec, AudioCodec::Aac);
        assert_eq!(info.audio_streams[0].sample_rate, 44100);
        assert_eq!(info.audio_streams[0].channels, 2);
        assert!(info.audio_streams[0].bitrate > 0);
    }
}
