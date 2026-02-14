//! MKV/WebM (Matroska) demuxer.
//!
//! Parses Matroska containers and yields video samples as `VideoPacket`s
//! with NAL data converted to Annex-B format, ready for hardware decoders
//! (NVDEC, Vulkan Video). Also yields audio packets for decoding.
//!
//! ## Supported codecs
//!
//! - Video: H.264 (V_MPEG4/ISO/AVC), H.265 (V_MPEGH/ISO/HEVC), VP9 (V_VP9), AV1 (V_AV1)
//! - Audio: AAC (A_AAC), Opus (A_OPUS), Vorbis (A_VORBIS), FLAC (A_FLAC)

pub mod cluster;
pub mod ebml;
pub mod elements;

use cluster::{parse_block, parse_simple_block, SimpleBlockInfo};
use ebml::{read_element, read_float, read_string, read_uint, skip_element};
use elements::*;
use ms_common::{
    AudioCodec, AudioPacket, AudioStreamInfo, ContainerInfo, DemuxError, PixelFormat, Rational,
    Resolution, TimeCode, VideoCodec, VideoPacket, VideoStreamInfo,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use tracing::{debug, info, trace, warn};

use crate::nal;
use crate::traits::Demuxer;

/// MKV/WebM demuxer — reads a Matroska file and yields video and audio packets.
pub struct MkvDemuxer {
    /// Path to the MKV/WebM file (kept for diagnostics/logging).
    #[allow(dead_code)]
    path: PathBuf,
    /// Buffered file reader.
    reader: BufReader<File>,
    /// Parsed track information.
    tracks: Vec<MkvTrackInfo>,
    /// TimecodeScale: nanoseconds per timecode tick (default 1_000_000 = 1ms).
    timecode_scale: u64,
    /// Segment duration in TimecodeScale units (if known).
    #[allow(dead_code)]
    duration: Option<f64>,
    /// Byte offset where the Segment data starts (after the Segment header).
    segment_data_offset: u64,
    /// Track number of the primary video track.
    video_track_num: Option<u64>,
    /// Track number of the primary audio track.
    audio_track_num: Option<u64>,
    /// Cue points for seeking.
    cues: Vec<MkvCuePoint>,
    /// Pre-built container info.
    container_info: ContainerInfo,
    /// Buffered video packets (read ahead from clusters).
    video_queue: VecDeque<VideoPacket>,
    /// Buffered audio packets (read ahead from clusters).
    audio_queue: VecDeque<AudioPacket>,
    /// Current cluster timecode (in TimecodeScale units).
    current_cluster_timecode: u64,
    /// Byte offset of the end of the current cluster (for sequential reading).
    /// `None` means no cluster is currently being read.
    cluster_end: Option<u64>,
    /// Whether we have reached the end of the stream.
    eos: bool,
}

impl MkvDemuxer {
    /// Open an MKV/WebM file and parse its structure (EBML header, Segment,
    /// Tracks, Info, optionally Cues).
    pub fn open(path: &Path) -> Result<Self, DemuxError> {
        info!("Opening MKV file: {}", path.display());

        let file = File::open(path).map_err(DemuxError::Io)?;
        let mut reader = BufReader::new(file);

        // Step 1: Verify EBML header
        verify_ebml_header(&mut reader)?;

        // Step 2: Find and enter Segment
        let segment_data_offset = find_segment(&mut reader)?;
        debug!("Segment data starts at offset {segment_data_offset}");

        // Step 3: Parse top-level Segment children (Info, Tracks, Cues)
        let mut timecode_scale: u64 = 1_000_000; // default
        let mut duration: Option<f64> = None;
        let mut tracks: Vec<MkvTrackInfo> = Vec::new();
        let mut cues: Vec<MkvCuePoint> = Vec::new();

        // We scan top-level Segment children until we find all metadata
        // or hit a Cluster.
        parse_segment_metadata(
            &mut reader,
            segment_data_offset,
            &mut timecode_scale,
            &mut duration,
            &mut tracks,
            &mut cues,
        )?;

        if tracks.is_empty() {
            return Err(DemuxError::InvalidStructure {
                offset: 0,
                reason: "No tracks found in Matroska file".into(),
            });
        }

        // Step 4: Identify primary video/audio tracks
        let video_track_num = tracks
            .iter()
            .find(|t| t.track_type == MkvTrackType::Video)
            .map(|t| t.track_number);

        if video_track_num.is_none() {
            return Err(DemuxError::NoVideoTrack);
        }

        let audio_track_num = tracks
            .iter()
            .find(|t| t.track_type == MkvTrackType::Audio)
            .map(|t| t.track_number);

        info!(
            "MKV: {} tracks, video=track#{:?}, audio=track#{:?}, timecode_scale={}, duration={:?}",
            tracks.len(),
            video_track_num,
            audio_track_num,
            timecode_scale,
            duration
        );

        // Step 5: Build container info
        let container_info =
            build_container_info(&tracks, timecode_scale, duration, segment_data_offset);

        Ok(MkvDemuxer {
            path: path.to_path_buf(),
            reader,
            tracks,
            timecode_scale,
            duration,
            segment_data_offset,
            video_track_num,
            audio_track_num,
            cues,
            container_info,
            video_queue: VecDeque::new(),
            audio_queue: VecDeque::new(),
            current_cluster_timecode: 0,
            cluster_end: None,
            eos: false,
        })
    }

    /// Get the video track info (if present).
    fn video_track(&self) -> Option<&MkvTrackInfo> {
        let vtn = self.video_track_num?;
        self.tracks.iter().find(|t| t.track_number == vtn)
    }

    /// Get the audio track info (if present).
    fn audio_track(&self) -> Option<&MkvTrackInfo> {
        let atn = self.audio_track_num?;
        self.tracks.iter().find(|t| t.track_number == atn)
    }

    /// Determine the VideoCodec from the codec ID string.
    fn video_codec(&self) -> VideoCodec {
        self.video_track()
            .map(|t| codec_id_to_video_codec(&t.codec_id))
            .unwrap_or(VideoCodec::H264)
    }

    /// Convert frame data to Annex-B format for H.264/H.265 codecs.
    ///
    /// For VP9 and AV1, data is passed through unchanged.
    fn convert_to_annexb(&self, block: &SimpleBlockInfo) -> Vec<u8> {
        let codec = self.video_codec();

        match codec {
            VideoCodec::H264 | VideoCodec::H265 => {
                // MKV stores H.264/H.265 frames in AVCC/HVCC format
                // (length-prefixed NALUs). We need to convert to Annex-B.
                let track = match self.video_track() {
                    Some(t) => t,
                    None => return block.frame_data.clone(),
                };

                // Determine NAL length size from CodecPrivate
                let length_size = track
                    .codec_private
                    .as_ref()
                    .and_then(|cp| avcc_nal_length_size(cp))
                    .unwrap_or(4);

                let annexb_data = nal::avcc_to_annexb(&block.frame_data, length_size);

                // Prepend SPS/PPS on keyframes
                if block.is_keyframe {
                    if let Some(cp) = &track.codec_private {
                        if codec == VideoCodec::H264 {
                            if let Some((sps, pps)) = extract_h264_sps_pps(cp) {
                                return nal::prepend_sps_pps(&sps, &pps, &annexb_data);
                            }
                        }
                        // For HEVC, the CodecPrivate is in HVCC format.
                        // Extract VPS/SPS/PPS and prepend.
                        if codec == VideoCodec::H265 {
                            if let Some(extra) = extract_hevc_parameter_sets(cp) {
                                let mut output =
                                    Vec::with_capacity(extra.len() + annexb_data.len());
                                output.extend_from_slice(&extra);
                                output.extend_from_slice(&annexb_data);
                                return output;
                            }
                        }
                    }
                }

                annexb_data
            }
            // VP9 and AV1 don't use Annex-B; pass through raw
            VideoCodec::Vp9 | VideoCodec::Av1 => block.frame_data.clone(),
        }
    }

    /// Read the next cluster's worth of blocks into the video/audio queues.
    ///
    /// Returns `true` if data was read, `false` if end of stream.
    fn fill_queues(&mut self) -> Result<bool, DemuxError> {
        if self.eos {
            return Ok(false);
        }

        loop {
            // If we're inside a cluster, read blocks until the cluster ends
            if let Some(cluster_end) = self.cluster_end {
                let pos = self.reader.stream_position().map_err(DemuxError::Io)?;
                if pos >= cluster_end {
                    // Cluster exhausted, look for next one
                    self.cluster_end = None;
                    continue;
                }

                // Read the next element within the cluster
                let elem = match read_element(&mut self.reader) {
                    Ok(e) => e,
                    Err(_) => {
                        self.eos = true;
                        return Ok(false);
                    }
                };

                match elem.id {
                    TIMECODE => {
                        self.current_cluster_timecode =
                            read_uint(&mut self.reader, elem.size)?;
                        trace!(
                            "Cluster timecode: {}",
                            self.current_cluster_timecode
                        );
                    }
                    SIMPLE_BLOCK => {
                        let data =
                            ebml::read_binary(&mut self.reader, elem.size)?;
                        self.process_simple_block(&data)?;

                        // Return once we've queued something useful
                        if !self.video_queue.is_empty() || !self.audio_queue.is_empty() {
                            return Ok(true);
                        }
                    }
                    BLOCK_GROUP => {
                        self.process_block_group(elem.size)?;

                        if !self.video_queue.is_empty() || !self.audio_queue.is_empty() {
                            return Ok(true);
                        }
                    }
                    _ => {
                        // Skip unknown elements within the cluster
                        if elem.size != u64::MAX {
                            skip_element(&mut self.reader, elem.size)?;
                        }
                    }
                }
            } else {
                // Look for the next Cluster at the top level of the Segment
                let elem = match read_element(&mut self.reader) {
                    Ok(e) => e,
                    Err(_) => {
                        self.eos = true;
                        return Ok(false);
                    }
                };

                match elem.id {
                    CLUSTER => {
                        // Enter the cluster
                        let end = if elem.size == u64::MAX {
                            // Unknown size: read until we hit another top-level element
                            None
                        } else {
                            Some(elem.data_offset() + elem.size)
                        };
                        self.cluster_end = end;
                        trace!(
                            "Entering cluster at offset {}, end={:?}",
                            elem.position,
                            end
                        );
                    }
                    CUES | SEEK_HEAD | INFO | TRACKS => {
                        // Skip metadata elements we've already parsed or don't need
                        if elem.size != u64::MAX {
                            skip_element(&mut self.reader, elem.size)?;
                        }
                    }
                    _ => {
                        // Skip unknown top-level elements
                        if elem.size != u64::MAX {
                            skip_element(&mut self.reader, elem.size)?;
                        } else {
                            // Can't skip unknown-size element; assume end
                            self.eos = true;
                            return Ok(false);
                        }
                    }
                }
            }
        }
    }

    /// Process a SimpleBlock and enqueue the resulting packet.
    fn process_simple_block(&mut self, data: &[u8]) -> Result<(), DemuxError> {
        let block = parse_simple_block(data)?;
        self.enqueue_block(block);
        Ok(())
    }

    /// Process a BlockGroup and enqueue the resulting packet.
    fn process_block_group(&mut self, group_size: u64) -> Result<(), DemuxError> {
        let group_end = self
            .reader
            .stream_position()
            .map_err(DemuxError::Io)?
            + group_size;

        let mut block_data: Option<Vec<u8>> = None;
        let mut has_reference = false;

        while self.reader.stream_position().map_err(DemuxError::Io)? < group_end {
            let elem = read_element(&mut self.reader)?;

            match elem.id {
                BLOCK => {
                    block_data = Some(ebml::read_binary(&mut self.reader, elem.size)?);
                }
                REFERENCE_BLOCK => {
                    has_reference = true;
                    // Read and discard the reference value
                    let _ref_val = ebml::read_sint(&mut self.reader, elem.size)?;
                }
                BLOCK_DURATION => {
                    // We don't use block duration currently
                    let _dur = read_uint(&mut self.reader, elem.size)?;
                }
                _ => {
                    if elem.size != u64::MAX {
                        skip_element(&mut self.reader, elem.size)?;
                    }
                }
            }
        }

        if let Some(data) = block_data {
            let block = parse_block(&data, has_reference)?;
            self.enqueue_block(block);
        }

        Ok(())
    }

    /// Convert a block into a VideoPacket or AudioPacket and enqueue it.
    fn enqueue_block(&mut self, block: SimpleBlockInfo) {
        // Compute absolute timecode in seconds
        let abs_timecode = self.current_cluster_timecode as i64 + block.timecode_offset as i64;
        let time_secs = timecode_to_secs(abs_timecode as u64, self.timecode_scale);

        if Some(block.track_number) == self.video_track_num {
            let data = self.convert_to_annexb(&block);
            self.video_queue.push_back(VideoPacket {
                data,
                pts: TimeCode(time_secs),
                dts: TimeCode(time_secs), // MKV doesn't have separate DTS
                is_keyframe: block.is_keyframe,
                codec: self.video_codec(),
            });
        } else if Some(block.track_number) == self.audio_track_num {
            if let Some(track) = self.audio_track() {
                let codec = codec_id_to_audio_codec(&track.codec_id);
                let (sample_rate, channels) = track
                    .audio
                    .as_ref()
                    .map(|a| (a.sampling_frequency as u32, a.channels as u16))
                    .unwrap_or((48000, 2));

                self.audio_queue.push_back(AudioPacket {
                    data: block.frame_data,
                    pts: TimeCode(time_secs),
                    codec,
                    sample_rate,
                    channels,
                });
            }
        }
        // Blocks for other tracks (subtitles etc.) are silently dropped.
    }
}

impl Demuxer for MkvDemuxer {
    fn probe(&self) -> &ContainerInfo {
        &self.container_info
    }

    fn next_video_packet(&mut self) -> Option<VideoPacket> {
        // Return queued packets first
        if let Some(pkt) = self.video_queue.pop_front() {
            return Some(pkt);
        }

        // Fill queues from the next cluster(s) until we get a video packet
        loop {
            match self.fill_queues() {
                Ok(true) => {
                    if let Some(pkt) = self.video_queue.pop_front() {
                        return Some(pkt);
                    }
                    // Got audio but no video; keep reading
                }
                Ok(false) | Err(_) => return None,
            }
        }
    }

    fn next_audio_packet(&mut self) -> Option<AudioPacket> {
        if let Some(pkt) = self.audio_queue.pop_front() {
            return Some(pkt);
        }

        loop {
            match self.fill_queues() {
                Ok(true) => {
                    if let Some(pkt) = self.audio_queue.pop_front() {
                        return Some(pkt);
                    }
                }
                Ok(false) | Err(_) => return None,
            }
        }
    }

    fn seek(&mut self, time_secs: f64) -> Result<(), DemuxError> {
        // Clear queues
        self.video_queue.clear();
        self.audio_queue.clear();
        self.cluster_end = None;
        self.eos = false;

        // If we have cue points, use them
        if let Some(video_track) = self.video_track_num {
            let target_timecode =
                secs_to_timecode(time_secs, self.timecode_scale);

            // Find the cue point at or before the target time for the video track
            let cue = self
                .cues
                .iter()
                .filter(|c| c.track == video_track && c.time <= target_timecode)
                .max_by_key(|c| c.time);

            if let Some(cue) = cue {
                let seek_offset = self.segment_data_offset + cue.cluster_position;
                debug!(
                    "Seeking to cue at time {} (target {:.3}s), cluster offset {}",
                    cue.time, time_secs, seek_offset
                );
                self.reader
                    .seek(SeekFrom::Start(seek_offset))
                    .map_err(DemuxError::Io)?;
                return Ok(());
            }
        }

        // No cues or no matching cue: scan from the beginning
        debug!(
            "No cue point found for {:.3}s, seeking from segment start",
            time_secs
        );
        self.reader
            .seek(SeekFrom::Start(self.segment_data_offset))
            .map_err(DemuxError::Io)?;

        // Scan clusters to find the right position
        let target_timecode = secs_to_timecode(time_secs, self.timecode_scale);
        let mut last_cluster_offset = self.segment_data_offset;

        loop {
            let pos = self.reader.stream_position().map_err(DemuxError::Io)?;
            let elem = match read_element(&mut self.reader) {
                Ok(e) => e,
                Err(_) => break,
            };

            if elem.id == CLUSTER {
                // Read the cluster timecode
                let inner = read_element(&mut self.reader)?;
                if inner.id == TIMECODE {
                    let tc = read_uint(&mut self.reader, inner.size)?;
                    if tc > target_timecode {
                        // We've gone past the target; seek to the previous cluster
                        break;
                    }
                    last_cluster_offset = pos;
                }
                // Skip to end of cluster
                if let Some(end) = elem.end_offset() {
                    self.reader
                        .seek(SeekFrom::Start(end))
                        .map_err(DemuxError::Io)?;
                } else {
                    break;
                }
            } else if elem.size != u64::MAX {
                skip_element(&mut self.reader, elem.size)?;
            } else {
                break;
            }
        }

        self.reader
            .seek(SeekFrom::Start(last_cluster_offset))
            .map_err(DemuxError::Io)?;
        Ok(())
    }

    fn has_audio(&self) -> bool {
        self.audio_track_num.is_some()
    }

    fn reset(&mut self) {
        self.video_queue.clear();
        self.audio_queue.clear();
        self.cluster_end = None;
        self.eos = false;
        self.current_cluster_timecode = 0;
        let _ = self.reader.seek(SeekFrom::Start(self.segment_data_offset));
    }
}

// ─── Parsing helpers ─────────────────────────────────────────────────

/// Verify the EBML header and check that the DocType is "matroska" or "webm".
fn verify_ebml_header<R: Read + Seek>(reader: &mut R) -> Result<(), DemuxError> {
    let elem = read_element(reader)?;

    if elem.id != EBML_HEADER {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!(
                "Expected EBML header (0x1A45DFA3), got 0x{:08X}",
                elem.id
            ),
        });
    }

    let header_end = elem.data_offset() + elem.size;
    let mut doc_type = String::new();

    while reader.stream_position().map_err(DemuxError::Io)? < header_end {
        let child = read_element(reader)?;

        match child.id {
            DOC_TYPE => {
                doc_type = read_string(reader, child.size)?;
                debug!("EBML DocType: {doc_type}");
            }
            _ => {
                if child.size != u64::MAX {
                    skip_element(reader, child.size)?;
                }
            }
        }
    }

    match doc_type.as_str() {
        "matroska" | "webm" => Ok(()),
        "" => Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "EBML header missing DocType".into(),
        }),
        other => Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!("Unsupported EBML DocType: \"{other}\""),
        }),
    }
}

/// Find the Segment element and return the byte offset where its data begins.
fn find_segment<R: Read + Seek>(reader: &mut R) -> Result<u64, DemuxError> {
    // The Segment should be the next top-level element after the EBML header.
    let elem = read_element(reader)?;

    if elem.id != SEGMENT {
        return Err(DemuxError::InvalidStructure {
            offset: elem.position,
            reason: format!(
                "Expected Segment (0x18538067), got 0x{:08X}",
                elem.id
            ),
        });
    }

    Ok(elem.data_offset())
}

/// Parse Segment metadata: Info, Tracks, and Cues.
///
/// Stops when the first Cluster is encountered (the reader is left positioned
/// at the start of the first Cluster element header).
fn parse_segment_metadata<R: Read + Seek>(
    reader: &mut R,
    _segment_data_offset: u64,
    timecode_scale: &mut u64,
    duration: &mut Option<f64>,
    tracks: &mut Vec<MkvTrackInfo>,
    cues: &mut Vec<MkvCuePoint>,
) -> Result<(), DemuxError> {
    loop {
        let pos = reader.stream_position().map_err(DemuxError::Io)?;
        let elem = match read_element(reader) {
            Ok(e) => e,
            Err(_) => return Ok(()), // End of file
        };

        match elem.id {
            INFO => {
                parse_info(reader, elem.size, timecode_scale, duration)?;
            }
            TRACKS => {
                *tracks = parse_tracks(reader, elem.size)?;
            }
            CUES => {
                *cues = parse_cues(reader, elem.size)?;
            }
            SEEK_HEAD => {
                // We could follow SeekHead pointers to find Cues, Tracks, etc.
                // For simplicity, just skip it and rely on sequential scanning.
                if elem.size != u64::MAX {
                    skip_element(reader, elem.size)?;
                }
            }
            CLUSTER => {
                // We've reached the first cluster. Seek back to its start
                // so fill_queues() can process it.
                reader.seek(SeekFrom::Start(pos)).map_err(DemuxError::Io)?;
                return Ok(());
            }
            _ => {
                if elem.size != u64::MAX {
                    skip_element(reader, elem.size)?;
                }
            }
        }
    }
}

/// Parse the Info element to extract TimecodeScale and Duration.
fn parse_info<R: Read + Seek>(
    reader: &mut R,
    size: u64,
    timecode_scale: &mut u64,
    duration: &mut Option<f64>,
) -> Result<(), DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        match child.id {
            TIMECODE_SCALE => {
                *timecode_scale = read_uint(reader, child.size)?;
                debug!("TimecodeScale: {}", *timecode_scale);
            }
            DURATION => {
                *duration = Some(read_float(reader, child.size)?);
                debug!("Duration: {:?}", *duration);
            }
            _ => {
                if child.size != u64::MAX {
                    skip_element(reader, child.size)?;
                }
            }
        }
    }

    Ok(())
}

/// Parse the Tracks element and return all track entries.
fn parse_tracks<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> Result<Vec<MkvTrackInfo>, DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;
    let mut tracks = Vec::new();

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        if child.id == TRACK_ENTRY {
            let track = parse_track_entry(reader, child.size)?;
            debug!(
                "Track #{}: type={:?}, codec={}",
                track.track_number, track.track_type, track.codec_id
            );
            tracks.push(track);
        } else if child.size != u64::MAX {
            skip_element(reader, child.size)?;
        }
    }

    Ok(tracks)
}

/// Parse a single TrackEntry element.
fn parse_track_entry<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> Result<MkvTrackInfo, DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;

    let mut track_number: u64 = 0;
    let mut track_type = MkvTrackType::Unknown(0);
    let mut codec_id = String::new();
    let mut codec_private: Option<Vec<u8>> = None;
    let mut default_duration_ns: Option<u64> = None;
    let mut video: Option<MkvVideoInfo> = None;
    let mut audio: Option<MkvAudioInfo> = None;

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        match child.id {
            TRACK_NUMBER => {
                track_number = read_uint(reader, child.size)?;
            }
            TRACK_TYPE => {
                let val = read_uint(reader, child.size)?;
                track_type = MkvTrackType::from_value(val);
            }
            CODEC_ID => {
                codec_id = read_string(reader, child.size)?;
            }
            CODEC_PRIVATE => {
                codec_private = Some(ebml::read_binary(reader, child.size)?);
            }
            DEFAULT_DURATION => {
                default_duration_ns = Some(read_uint(reader, child.size)?);
            }
            VIDEO => {
                video = Some(parse_video_settings(reader, child.size)?);
            }
            AUDIO => {
                audio = Some(parse_audio_settings(reader, child.size)?);
            }
            _ => {
                if child.size != u64::MAX {
                    skip_element(reader, child.size)?;
                }
            }
        }
    }

    Ok(MkvTrackInfo {
        track_number,
        track_type,
        codec_id,
        codec_private,
        default_duration_ns,
        video,
        audio,
    })
}

/// Parse a Video settings sub-element.
fn parse_video_settings<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> Result<MkvVideoInfo, DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;

    let mut pixel_width: u32 = 0;
    let mut pixel_height: u32 = 0;
    let mut display_width: Option<u32> = None;
    let mut display_height: Option<u32> = None;

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        match child.id {
            PIXEL_WIDTH => {
                pixel_width = read_uint(reader, child.size)? as u32;
            }
            PIXEL_HEIGHT => {
                pixel_height = read_uint(reader, child.size)? as u32;
            }
            DISPLAY_WIDTH => {
                display_width = Some(read_uint(reader, child.size)? as u32);
            }
            DISPLAY_HEIGHT => {
                display_height = Some(read_uint(reader, child.size)? as u32);
            }
            _ => {
                if child.size != u64::MAX {
                    skip_element(reader, child.size)?;
                }
            }
        }
    }

    Ok(MkvVideoInfo {
        pixel_width,
        pixel_height,
        display_width,
        display_height,
    })
}

/// Parse an Audio settings sub-element.
fn parse_audio_settings<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> Result<MkvAudioInfo, DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;

    let mut sampling_frequency: f64 = 8000.0; // default per spec
    let mut output_sampling_frequency: Option<f64> = None;
    let mut channels: u32 = 1; // default per spec
    let mut bit_depth: Option<u32> = None;

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        match child.id {
            SAMPLING_FREQUENCY => {
                sampling_frequency = read_float(reader, child.size)?;
            }
            OUTPUT_SAMPLING_FREQUENCY => {
                output_sampling_frequency = Some(read_float(reader, child.size)?);
            }
            CHANNELS => {
                channels = read_uint(reader, child.size)? as u32;
            }
            BIT_DEPTH => {
                bit_depth = Some(read_uint(reader, child.size)? as u32);
            }
            _ => {
                if child.size != u64::MAX {
                    skip_element(reader, child.size)?;
                }
            }
        }
    }

    Ok(MkvAudioInfo {
        sampling_frequency,
        output_sampling_frequency,
        channels,
        bit_depth,
    })
}

/// Parse the Cues element (seek index).
fn parse_cues<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> Result<Vec<MkvCuePoint>, DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;
    let mut cues = Vec::new();

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        if child.id == CUE_POINT {
            if let Ok(cue) = parse_cue_point(reader, child.size) {
                cues.push(cue);
            }
        } else if child.size != u64::MAX {
            skip_element(reader, child.size)?;
        }
    }

    debug!("Parsed {} cue points", cues.len());
    Ok(cues)
}

/// Parse a single CuePoint element.
fn parse_cue_point<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> Result<MkvCuePoint, DemuxError> {
    let end = reader.stream_position().map_err(DemuxError::Io)? + size;

    let mut time: u64 = 0;
    let mut track: u64 = 0;
    let mut cluster_position: u64 = 0;
    let mut relative_position: Option<u64> = None;

    while reader.stream_position().map_err(DemuxError::Io)? < end {
        let child = read_element(reader)?;

        match child.id {
            CUE_TIME => {
                time = read_uint(reader, child.size)?;
            }
            CUE_TRACK_POSITIONS => {
                let tp_end =
                    reader.stream_position().map_err(DemuxError::Io)? + child.size;

                while reader.stream_position().map_err(DemuxError::Io)? < tp_end {
                    let inner = read_element(reader)?;

                    match inner.id {
                        CUE_TRACK => {
                            track = read_uint(reader, inner.size)?;
                        }
                        CUE_CLUSTER_POSITION => {
                            cluster_position = read_uint(reader, inner.size)?;
                        }
                        CUE_RELATIVE_POSITION => {
                            relative_position =
                                Some(read_uint(reader, inner.size)?);
                        }
                        _ => {
                            if inner.size != u64::MAX {
                                skip_element(reader, inner.size)?;
                            }
                        }
                    }
                }
            }
            _ => {
                if child.size != u64::MAX {
                    skip_element(reader, child.size)?;
                }
            }
        }
    }

    Ok(MkvCuePoint {
        time,
        track,
        cluster_position,
        relative_position,
    })
}

// ─── Codec helpers ───────────────────────────────────────────────────

/// Map a Matroska codec ID string to our VideoCodec enum.
fn codec_id_to_video_codec(codec_id: &str) -> VideoCodec {
    match codec_id {
        "V_MPEG4/ISO/AVC" => VideoCodec::H264,
        "V_MPEGH/ISO/HEVC" => VideoCodec::H265,
        "V_VP9" => VideoCodec::Vp9,
        "V_AV1" => VideoCodec::Av1,
        other => {
            warn!("Unknown MKV video codec: {other}, defaulting to H264");
            VideoCodec::H264
        }
    }
}

/// Map a Matroska codec ID string to our AudioCodec enum.
fn codec_id_to_audio_codec(codec_id: &str) -> AudioCodec {
    match codec_id {
        "A_AAC" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG4/LC" | "A_AAC/MPEG4/LC/SBR" => AudioCodec::Aac,
        "A_OPUS" => AudioCodec::Opus,
        "A_VORBIS" => AudioCodec::Vorbis,
        "A_FLAC" => AudioCodec::Flac,
        "A_MPEG/L3" => AudioCodec::Mp3,
        other => {
            warn!("Unknown MKV audio codec: {other}, defaulting to Aac");
            AudioCodec::Aac
        }
    }
}

/// Extract NAL length size from an AVCC (H.264) CodecPrivate blob.
///
/// The AVCC header layout is:
/// ```text
/// [0] configuration_version = 1
/// [1] avc_profile_indication
/// [2] profile_compatibility
/// [3] avc_level_indication
/// [4] xxxxxx(length_size_minus_one & 0x03)
/// ```
fn avcc_nal_length_size(codec_private: &[u8]) -> Option<u8> {
    if codec_private.len() < 7 {
        return None;
    }
    // First byte should be 1 (configurationVersion)
    if codec_private[0] != 1 {
        return None;
    }
    Some((codec_private[4] & 0x03) + 1)
}

/// Extract SPS and PPS from an AVCC CodecPrivate blob.
///
/// Returns `(first_sps, first_pps)` if both are present.
fn extract_h264_sps_pps(codec_private: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    if codec_private.len() < 8 || codec_private[0] != 1 {
        return None;
    }

    let mut offset = 5;

    // Number of SPS
    let num_sps = (codec_private[offset] & 0x1F) as usize;
    offset += 1;

    let mut sps_list = Vec::new();
    for _ in 0..num_sps {
        if offset + 2 > codec_private.len() {
            return None;
        }
        let sps_len =
            u16::from_be_bytes([codec_private[offset], codec_private[offset + 1]]) as usize;
        offset += 2;

        if offset + sps_len > codec_private.len() {
            return None;
        }
        sps_list.push(codec_private[offset..offset + sps_len].to_vec());
        offset += sps_len;
    }

    // Number of PPS
    if offset >= codec_private.len() {
        return None;
    }
    let num_pps = codec_private[offset] as usize;
    offset += 1;

    let mut pps_list = Vec::new();
    for _ in 0..num_pps {
        if offset + 2 > codec_private.len() {
            return None;
        }
        let pps_len =
            u16::from_be_bytes([codec_private[offset], codec_private[offset + 1]]) as usize;
        offset += 2;

        if offset + pps_len > codec_private.len() {
            return None;
        }
        pps_list.push(codec_private[offset..offset + pps_len].to_vec());
        offset += pps_len;
    }

    if sps_list.is_empty() || pps_list.is_empty() {
        return None;
    }

    Some((sps_list[0].clone(), pps_list[0].clone()))
}

/// Extract VPS/SPS/PPS from an HEVC CodecPrivate (HVCC format) and
/// return them as Annex-B-formatted data.
fn extract_hevc_parameter_sets(codec_private: &[u8]) -> Option<Vec<u8>> {
    // HVCC format (ISO 14496-15):
    // Bytes 0-21: general configuration
    // Byte 21: length_size_minus_one & 0x03
    // Byte 22: num_of_arrays
    // Then arrays of NAL units...
    if codec_private.len() < 23 {
        return None;
    }

    let num_arrays = codec_private[22] as usize;
    let mut offset = 23;
    let mut output = Vec::new();

    for _ in 0..num_arrays {
        if offset + 3 > codec_private.len() {
            break;
        }
        // Byte: array_completeness(1) | reserved(1) | NAL_unit_type(6)
        let _nal_type = codec_private[offset] & 0x3F;
        offset += 1;

        let num_nalus =
            u16::from_be_bytes([codec_private[offset], codec_private[offset + 1]]) as usize;
        offset += 2;

        for _ in 0..num_nalus {
            if offset + 2 > codec_private.len() {
                break;
            }
            let nalu_len = u16::from_be_bytes([codec_private[offset], codec_private[offset + 1]])
                as usize;
            offset += 2;

            if offset + nalu_len > codec_private.len() {
                break;
            }

            output.extend_from_slice(&nal::ANNEXB_START_CODE);
            output.extend_from_slice(&codec_private[offset..offset + nalu_len]);
            offset += nalu_len;
        }
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

// ─── Time conversion helpers ─────────────────────────────────────────

/// Convert a timecode (in TimecodeScale units) to seconds.
fn timecode_to_secs(timecode: u64, timecode_scale: u64) -> f64 {
    (timecode as f64 * timecode_scale as f64) / 1_000_000_000.0
}

/// Convert seconds to a timecode (in TimecodeScale units).
fn secs_to_timecode(secs: f64, timecode_scale: u64) -> u64 {
    if timecode_scale == 0 {
        return 0;
    }
    ((secs * 1_000_000_000.0) / timecode_scale as f64) as u64
}

// ─── Container info builder ──────────────────────────────────────────

/// Build `ContainerInfo` from parsed MKV metadata.
fn build_container_info(
    tracks: &[MkvTrackInfo],
    timecode_scale: u64,
    duration: Option<f64>,
    _segment_data_offset: u64,
) -> ContainerInfo {
    let duration_secs = duration
        .map(|d| (d * timecode_scale as f64) / 1_000_000_000.0)
        .unwrap_or(0.0);

    let video_streams: Vec<VideoStreamInfo> = tracks
        .iter()
        .filter(|t| t.track_type == MkvTrackType::Video)
        .map(|track| {
            let codec = codec_id_to_video_codec(&track.codec_id);
            let (width, height) = track
                .video
                .as_ref()
                .map(|v| (v.pixel_width, v.pixel_height))
                .unwrap_or((0, 0));

            // Estimate FPS from DefaultDuration (nanoseconds per frame)
            let fps = track
                .default_duration_ns
                .filter(|&d| d > 0)
                .map(|d| {
                    let fps_f64 = 1_000_000_000.0 / d as f64;
                    // Try to match common frame rates
                    match_common_fps(fps_f64)
                })
                .unwrap_or(Rational::FPS_30);

            // Build extra_data (SPS/PPS in Annex-B format)
            let extra_data = track
                .codec_private
                .as_ref()
                .and_then(|cp| match codec {
                    VideoCodec::H264 => {
                        let (sps, pps) = extract_h264_sps_pps(cp)?;
                        let mut data = Vec::new();
                        data.extend_from_slice(&nal::ANNEXB_START_CODE);
                        data.extend_from_slice(&sps);
                        data.extend_from_slice(&nal::ANNEXB_START_CODE);
                        data.extend_from_slice(&pps);
                        Some(data)
                    }
                    VideoCodec::H265 => extract_hevc_parameter_sets(cp),
                    _ => Some(cp.clone()),
                })
                .unwrap_or_default();

            VideoStreamInfo {
                codec,
                resolution: Resolution::new(width, height),
                fps,
                duration: TimeCode(duration_secs),
                bitrate: 0, // MKV doesn't store bitrate; would need to be computed
                pixel_format: PixelFormat::Nv12,
                extra_data,
            }
        })
        .collect();

    let audio_streams: Vec<AudioStreamInfo> = tracks
        .iter()
        .filter(|t| t.track_type == MkvTrackType::Audio)
        .map(|track| {
            let codec = codec_id_to_audio_codec(&track.codec_id);
            let (sample_rate, channels) = track
                .audio
                .as_ref()
                .map(|a| (a.sampling_frequency as u32, a.channels as u16))
                .unwrap_or((48000, 2));

            AudioStreamInfo {
                codec,
                sample_rate,
                channels,
                duration: TimeCode(duration_secs),
                bitrate: 0,
            }
        })
        .collect();

    ContainerInfo {
        video_streams,
        audio_streams,
        duration: TimeCode(duration_secs),
    }
}

/// Try to match a floating-point FPS value to a common Rational representation.
fn match_common_fps(fps: f64) -> Rational {
    const COMMON: [(f64, Rational); 8] = [
        (23.976, Rational { num: 24000, den: 1001 }),
        (24.0, Rational::FPS_24),
        (25.0, Rational::FPS_25),
        (29.97, Rational { num: 30000, den: 1001 }),
        (30.0, Rational::FPS_30),
        (50.0, Rational { num: 50, den: 1 }),
        (59.94, Rational { num: 60000, den: 1001 }),
        (60.0, Rational::FPS_60),
    ];

    let mut best_match: Option<(f64, Rational)> = None;
    for (target, rational) in &COMMON {
        let diff = (fps - target).abs();
        if diff < 0.05 {
            if best_match.is_none() || diff < best_match.unwrap().0 {
                best_match = Some((diff, *rational));
            }
        }
    }
    if let Some((_, rational)) = best_match {
        return rational;
    }

    // Fallback: approximate as integer fraction
    let num = (fps * 1000.0).round() as u32;
    Rational::new(num, 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_id_to_video_codec() {
        assert_eq!(
            codec_id_to_video_codec("V_MPEG4/ISO/AVC"),
            VideoCodec::H264
        );
        assert_eq!(
            codec_id_to_video_codec("V_MPEGH/ISO/HEVC"),
            VideoCodec::H265
        );
        assert_eq!(codec_id_to_video_codec("V_VP9"), VideoCodec::Vp9);
        assert_eq!(codec_id_to_video_codec("V_AV1"), VideoCodec::Av1);
    }

    #[test]
    fn test_codec_id_to_audio_codec() {
        assert_eq!(codec_id_to_audio_codec("A_AAC"), AudioCodec::Aac);
        assert_eq!(
            codec_id_to_audio_codec("A_AAC/MPEG4/LC"),
            AudioCodec::Aac
        );
        assert_eq!(codec_id_to_audio_codec("A_OPUS"), AudioCodec::Opus);
        assert_eq!(codec_id_to_audio_codec("A_VORBIS"), AudioCodec::Vorbis);
        assert_eq!(codec_id_to_audio_codec("A_FLAC"), AudioCodec::Flac);
        assert_eq!(codec_id_to_audio_codec("A_MPEG/L3"), AudioCodec::Mp3);
    }

    #[test]
    fn test_timecode_to_secs() {
        // Default scale: 1_000_000 ns/tick
        // 5000 ticks * 1_000_000 ns/tick = 5_000_000_000 ns = 5.0 seconds
        let secs = timecode_to_secs(5000, 1_000_000);
        assert!((secs - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_timecode_to_secs_custom_scale() {
        // Scale: 500_000 ns/tick (0.5ms)
        // 10000 ticks * 500_000 = 5_000_000_000 ns = 5.0 seconds
        let secs = timecode_to_secs(10000, 500_000);
        assert!((secs - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_secs_to_timecode() {
        let tc = secs_to_timecode(5.0, 1_000_000);
        assert_eq!(tc, 5000);
    }

    #[test]
    fn test_secs_to_timecode_zero_scale() {
        let tc = secs_to_timecode(5.0, 0);
        assert_eq!(tc, 0);
    }

    #[test]
    fn test_match_common_fps() {
        let fps_30 = match_common_fps(30.0);
        assert_eq!(fps_30.num, 30);
        assert_eq!(fps_30.den, 1);

        let fps_29_97 = match_common_fps(29.97);
        assert_eq!(fps_29_97.num, 30000);
        assert_eq!(fps_29_97.den, 1001);

        let fps_23_976 = match_common_fps(23.976);
        assert_eq!(fps_23_976.num, 24000);
        assert_eq!(fps_23_976.den, 1001);

        let fps_60 = match_common_fps(60.0);
        assert_eq!(fps_60.num, 60);
        assert_eq!(fps_60.den, 1);
    }

    #[test]
    fn test_match_common_fps_unusual() {
        // 15 fps -> approximate as 15000/1000
        let fps = match_common_fps(15.0);
        assert_eq!(fps.num, 15000);
        assert_eq!(fps.den, 1000);
    }

    #[test]
    fn test_avcc_nal_length_size() {
        // Minimal valid AVCC: version=1, profile, compat, level, length_size_minus_one=3
        let mut cp = vec![0x01, 0x64, 0x00, 0x1F, 0xFF]; // 0xFF & 0x03 = 3, +1 = 4
        cp.extend_from_slice(&[0xE1, 0x00]); // needed for len >= 7
        assert_eq!(avcc_nal_length_size(&cp), Some(4));

        // length_size_minus_one = 1 -> length_size = 2
        cp[4] = 0xFD; // 0xFD & 0x03 = 1
        assert_eq!(avcc_nal_length_size(&cp), Some(2));
    }

    #[test]
    fn test_avcc_nal_length_size_too_short() {
        let cp = vec![0x01, 0x64, 0x00];
        assert_eq!(avcc_nal_length_size(&cp), None);
    }

    #[test]
    fn test_avcc_nal_length_size_wrong_version() {
        let cp = vec![0x02, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00];
        assert_eq!(avcc_nal_length_size(&cp), None);
    }

    #[test]
    fn test_extract_h264_sps_pps() {
        // Build a minimal AVCC with 1 SPS (3 bytes) and 1 PPS (2 bytes)
        let mut cp = Vec::new();
        cp.push(0x01); // configurationVersion
        cp.push(0x64); // profile
        cp.push(0x00); // compat
        cp.push(0x1F); // level
        cp.push(0xFF); // length_size_minus_one = 3
        cp.push(0xE1); // num_sps = 1 (0xE1 & 0x1F = 1)

        // SPS: 3 bytes
        cp.extend_from_slice(&[0x00, 0x03]); // sps_len = 3
        cp.extend_from_slice(&[0x67, 0x42, 0xC0]); // SPS data

        cp.push(0x01); // num_pps = 1

        // PPS: 2 bytes
        cp.extend_from_slice(&[0x00, 0x02]); // pps_len = 2
        cp.extend_from_slice(&[0x68, 0xCE]); // PPS data

        let (sps, pps) = extract_h264_sps_pps(&cp).unwrap();
        assert_eq!(sps, vec![0x67, 0x42, 0xC0]);
        assert_eq!(pps, vec![0x68, 0xCE]);
    }

    #[test]
    fn test_extract_h264_sps_pps_empty() {
        // Version != 1
        let cp = vec![0x00; 20];
        assert!(extract_h264_sps_pps(&cp).is_none());
    }

    #[test]
    fn test_extract_hevc_parameter_sets() {
        // Build a minimal HVCC with 1 array containing 1 NALU (4 bytes)
        let mut cp = vec![0u8; 22]; // 22 bytes of general configuration
        cp.push(1); // num_of_arrays = 1

        // Array: type=32 (VPS), 1 NALU
        cp.push(0x20); // array_completeness=0, nal_type=32
        cp.extend_from_slice(&[0x00, 0x01]); // num_nalus = 1
        cp.extend_from_slice(&[0x00, 0x04]); // nalu_len = 4
        cp.extend_from_slice(&[0x40, 0x01, 0x0C, 0x01]); // VPS data

        let result = extract_hevc_parameter_sets(&cp).unwrap();
        // Should be: start_code + VPS data
        assert_eq!(&result[..4], &nal::ANNEXB_START_CODE);
        assert_eq!(&result[4..], &[0x40, 0x01, 0x0C, 0x01]);
    }

    #[test]
    fn test_extract_hevc_parameter_sets_too_short() {
        let cp = vec![0u8; 10];
        assert!(extract_hevc_parameter_sets(&cp).is_none());
    }

    #[test]
    fn test_verify_ebml_header_valid() {
        // Construct a minimal EBML header with DocType = "matroska"
        let mut data = Vec::new();

        // EBML Header ID: 0x1A45DFA3
        data.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);

        // Build the header content first to know its size
        let mut content = Vec::new();
        // DocType element: ID=0x4282, size=8, "matroska"
        content.extend_from_slice(&[0x42, 0x82]); // DocType ID (2-byte)
        content.push(0x88); // size = 8
        content.extend_from_slice(b"matroska");

        // Header size as VINT
        let content_len = content.len() as u8;
        data.push(0x80 | content_len); // 1-byte VINT size

        data.extend_from_slice(&content);

        let mut cursor = std::io::Cursor::new(data);
        assert!(verify_ebml_header(&mut cursor).is_ok());
    }

    #[test]
    fn test_verify_ebml_header_webm() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);

        let mut content = Vec::new();
        content.extend_from_slice(&[0x42, 0x82]);
        content.push(0x84); // size = 4
        content.extend_from_slice(b"webm");

        let content_len = content.len() as u8;
        data.push(0x80 | content_len);
        data.extend_from_slice(&content);

        let mut cursor = std::io::Cursor::new(data);
        assert!(verify_ebml_header(&mut cursor).is_ok());
    }

    #[test]
    fn test_verify_ebml_header_invalid_doctype() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);

        let mut content = Vec::new();
        content.extend_from_slice(&[0x42, 0x82]);
        content.push(0x83); // size = 3
        content.extend_from_slice(b"avi");

        let content_len = content.len() as u8;
        data.push(0x80 | content_len);
        data.extend_from_slice(&content);

        let mut cursor = std::io::Cursor::new(data);
        assert!(verify_ebml_header(&mut cursor).is_err());
    }

    #[test]
    fn test_verify_ebml_header_not_ebml() {
        // Start with wrong ID
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x80];
        let mut cursor = std::io::Cursor::new(data);
        assert!(verify_ebml_header(&mut cursor).is_err());
    }

    #[test]
    fn test_build_container_info_video_only() {
        let tracks = vec![MkvTrackInfo {
            track_number: 1,
            track_type: MkvTrackType::Video,
            codec_id: "V_MPEG4/ISO/AVC".to_string(),
            codec_private: None,
            default_duration_ns: Some(33_333_333), // ~30fps
            video: Some(MkvVideoInfo {
                pixel_width: 1920,
                pixel_height: 1080,
                display_width: None,
                display_height: None,
            }),
            audio: None,
        }];

        let info = build_container_info(&tracks, 1_000_000, Some(10000.0), 0);

        assert_eq!(info.video_streams.len(), 1);
        assert_eq!(info.audio_streams.len(), 0);
        assert_eq!(info.video_streams[0].codec, VideoCodec::H264);
        assert_eq!(info.video_streams[0].resolution.width, 1920);
        assert_eq!(info.video_streams[0].resolution.height, 1080);
        assert_eq!(info.video_streams[0].fps.num, 30);
        assert_eq!(info.video_streams[0].fps.den, 1);
        // Duration: 10000 * 1_000_000 / 1_000_000_000 = 10.0s
        assert!((info.duration.0 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_container_info_video_and_audio() {
        let tracks = vec![
            MkvTrackInfo {
                track_number: 1,
                track_type: MkvTrackType::Video,
                codec_id: "V_VP9".to_string(),
                codec_private: None,
                default_duration_ns: Some(41_708_333), // ~24fps (23.976)
                video: Some(MkvVideoInfo {
                    pixel_width: 3840,
                    pixel_height: 2160,
                    display_width: None,
                    display_height: None,
                }),
                audio: None,
            },
            MkvTrackInfo {
                track_number: 2,
                track_type: MkvTrackType::Audio,
                codec_id: "A_OPUS".to_string(),
                codec_private: None,
                default_duration_ns: None,
                video: None,
                audio: Some(MkvAudioInfo {
                    sampling_frequency: 48000.0,
                    output_sampling_frequency: None,
                    channels: 2,
                    bit_depth: None,
                }),
            },
        ];

        let info = build_container_info(&tracks, 1_000_000, Some(60000.0), 0);

        assert_eq!(info.video_streams.len(), 1);
        assert_eq!(info.audio_streams.len(), 1);
        assert_eq!(info.video_streams[0].codec, VideoCodec::Vp9);
        assert_eq!(info.audio_streams[0].codec, AudioCodec::Opus);
        assert_eq!(info.audio_streams[0].sample_rate, 48000);
        assert_eq!(info.audio_streams[0].channels, 2);
    }
}
