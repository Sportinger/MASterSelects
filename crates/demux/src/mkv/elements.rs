//! Matroska element IDs and parsed structures.
//!
//! Defines constants for all relevant Matroska/WebM element IDs and
//! data structures for parsed track information.

// ─── EBML Header ─────────────────────────────────────────────────────

/// EBML Header element (container).
pub const EBML_HEADER: u32 = 0x1A45DFA3;
/// EBML Version.
pub const EBML_VERSION: u32 = 0x4286;
/// EBML Read Version.
pub const EBML_READ_VERSION: u32 = 0x42F7;
/// EBML Max ID Length.
pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
/// EBML Max Size Length.
pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
/// Document type string (e.g., "matroska", "webm").
pub const DOC_TYPE: u32 = 0x4282;
/// Document type version.
pub const DOC_TYPE_VERSION: u32 = 0x4287;
/// Document type read version.
pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;

// ─── Segment ─────────────────────────────────────────────────────────

/// Segment (top-level container for all data).
pub const SEGMENT: u32 = 0x18538067;

// ─── Meta Seek Information (SeekHead) ────────────────────────────────

/// SeekHead: contains position hints for top-level elements.
pub const SEEK_HEAD: u32 = 0x114D9B74;
/// A single Seek entry in SeekHead.
pub const SEEK: u32 = 0x4DBB;
/// SeekID: the ID of the element being located.
pub const SEEK_ID: u32 = 0x53AB;
/// SeekPosition: byte offset from start of Segment.
pub const SEEK_POSITION: u32 = 0x53AC;

// ─── Segment Information ─────────────────────────────────────────────

/// Info element (segment information).
pub const INFO: u32 = 0x1549A966;
/// TimecodeScale: nanoseconds per timecode tick (default 1_000_000 = 1ms).
pub const TIMECODE_SCALE: u32 = 0x2AD7B1;
/// Duration: total segment duration in TimecodeScale units (float).
pub const DURATION: u32 = 0x4489;
/// Muxing application name.
pub const MUXING_APP: u32 = 0x4D80;
/// Writing application name.
pub const WRITING_APP: u32 = 0x5741;

// ─── Track Information ───────────────────────────────────────────────

/// Tracks element (container for all track entries).
pub const TRACKS: u32 = 0x1654AE6B;
/// A single track entry.
pub const TRACK_ENTRY: u32 = 0xAE;
/// Track number (used in SimpleBlock/Block to identify the track).
pub const TRACK_NUMBER: u32 = 0xD7;
/// Track UID (globally unique, not used for block references).
pub const TRACK_UID: u32 = 0x73C5;
/// Track type (1=video, 2=audio, 17=subtitle).
pub const TRACK_TYPE: u32 = 0x83;
/// Codec ID string (e.g., "V_MPEG4/ISO/AVC").
pub const CODEC_ID: u32 = 0x86;
/// Codec-private data (e.g., SPS/PPS for H.264, codec init data).
pub const CODEC_PRIVATE: u32 = 0x63A2;
/// Default duration of a frame in nanoseconds.
pub const DEFAULT_DURATION: u32 = 0x23E383;
/// Track language.
pub const LANGUAGE: u32 = 0x22B59C;
/// Track name / description.
pub const TRACK_NAME: u32 = 0x536E;
/// Whether the track should be active by default.
pub const FLAG_DEFAULT: u32 = 0x88;
/// Whether the track is enabled.
pub const FLAG_ENABLED: u32 = 0xB9;
/// Codec delay in nanoseconds.
pub const CODEC_DELAY: u32 = 0x56AA;
/// Seek pre-roll in nanoseconds.
pub const SEEK_PRE_ROLL: u32 = 0x56BB;

// ─── Video Settings ──────────────────────────────────────────────────

/// Video settings sub-element within a TrackEntry.
pub const VIDEO: u32 = 0xE0;
/// Pixel width.
pub const PIXEL_WIDTH: u32 = 0xB0;
/// Pixel height.
pub const PIXEL_HEIGHT: u32 = 0xBA;
/// Display width (for DAR calculation; optional).
pub const DISPLAY_WIDTH: u32 = 0x54B0;
/// Display height (for DAR calculation; optional).
pub const DISPLAY_HEIGHT: u32 = 0x54BA;

// ─── Audio Settings ──────────────────────────────────────────────────

/// Audio settings sub-element within a TrackEntry.
pub const AUDIO: u32 = 0xE1;
/// Sampling frequency in Hz.
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
/// Output sampling frequency (for SBR; optional).
pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
/// Number of audio channels.
pub const CHANNELS: u32 = 0x9F;
/// Bit depth per sample (optional).
pub const BIT_DEPTH: u32 = 0x6264;

// ─── Cluster ─────────────────────────────────────────────────────────

/// Cluster element (container for frames).
pub const CLUSTER: u32 = 0x1F43B675;
/// Cluster timecode in TimecodeScale units.
pub const TIMECODE: u32 = 0xE7;
/// SimpleBlock: track + timecode offset + flags + frame data.
pub const SIMPLE_BLOCK: u32 = 0xA3;
/// BlockGroup container.
pub const BLOCK_GROUP: u32 = 0xA0;
/// Block within a BlockGroup.
pub const BLOCK: u32 = 0xA1;
/// Block duration in TimecodeScale units (within BlockGroup).
pub const BLOCK_DURATION: u32 = 0x9B;
/// Reference block (negative = reference to previous, used for P/B frames).
pub const REFERENCE_BLOCK: u32 = 0xFB;

// ─── Cues (Seek Index) ──────────────────────────────────────────────

/// Cues element (seek index, like MP4's stss).
pub const CUES: u32 = 0x1C53BB6B;
/// A single cue point.
pub const CUE_POINT: u32 = 0xBB;
/// Cue time in TimecodeScale units.
pub const CUE_TIME: u32 = 0xB3;
/// Track positions for a cue point.
pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
/// Track number this cue refers to.
pub const CUE_TRACK: u32 = 0xF7;
/// Cluster position (byte offset from Segment start).
pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
/// Relative position within the cluster.
pub const CUE_RELATIVE_POSITION: u32 = 0xF0;

// ─── Parsed Structures ──────────────────────────────────────────────

/// The type of a Matroska track.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MkvTrackType {
    Video,
    Audio,
    Subtitle,
    Unknown(u64),
}

impl MkvTrackType {
    /// Parse from the TrackType element value.
    pub fn from_value(val: u64) -> Self {
        match val {
            1 => Self::Video,
            2 => Self::Audio,
            17 => Self::Subtitle,
            other => Self::Unknown(other),
        }
    }
}

/// Parsed video-specific information from a MKV TrackEntry.
#[derive(Clone, Debug)]
pub struct MkvVideoInfo {
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub display_width: Option<u32>,
    pub display_height: Option<u32>,
}

/// Parsed audio-specific information from a MKV TrackEntry.
#[derive(Clone, Debug)]
pub struct MkvAudioInfo {
    pub sampling_frequency: f64,
    pub output_sampling_frequency: Option<f64>,
    pub channels: u32,
    pub bit_depth: Option<u32>,
}

/// Parsed track information from a Matroska TrackEntry.
#[derive(Clone, Debug)]
pub struct MkvTrackInfo {
    /// The track number (used in SimpleBlock to identify the track).
    pub track_number: u64,
    /// Track type (video, audio, subtitle).
    pub track_type: MkvTrackType,
    /// Codec ID string (e.g., "V_MPEG4/ISO/AVC", "A_AAC", "A_OPUS").
    pub codec_id: String,
    /// Codec-private data (SPS/PPS for H.264, etc.).
    pub codec_private: Option<Vec<u8>>,
    /// Default frame duration in nanoseconds (if specified).
    pub default_duration_ns: Option<u64>,
    /// Video-specific info (if track_type == Video).
    pub video: Option<MkvVideoInfo>,
    /// Audio-specific info (if track_type == Audio).
    pub audio: Option<MkvAudioInfo>,
}

/// A cue point from the Cues element (used for seeking).
#[derive(Clone, Debug)]
pub struct MkvCuePoint {
    /// Time in TimecodeScale units.
    pub time: u64,
    /// Track number this cue applies to.
    pub track: u64,
    /// Byte offset of the cluster from the start of the Segment data.
    pub cluster_position: u64,
    /// Relative position within the cluster (optional).
    pub relative_position: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_id_constants() {
        assert_eq!(EBML_HEADER, 0x1A45DFA3);
        assert_eq!(SEGMENT, 0x18538067);
        assert_eq!(TRACKS, 0x1654AE6B);
        assert_eq!(CLUSTER, 0x1F43B675);
        assert_eq!(SIMPLE_BLOCK, 0xA3);
        assert_eq!(CUES, 0x1C53BB6B);
        assert_eq!(INFO, 0x1549A966);
        assert_eq!(TIMECODE_SCALE, 0x2AD7B1);
    }

    #[test]
    fn test_track_type_from_value() {
        assert_eq!(MkvTrackType::from_value(1), MkvTrackType::Video);
        assert_eq!(MkvTrackType::from_value(2), MkvTrackType::Audio);
        assert_eq!(MkvTrackType::from_value(17), MkvTrackType::Subtitle);
        assert_eq!(MkvTrackType::from_value(99), MkvTrackType::Unknown(99));
    }

    #[test]
    fn test_mkv_video_info() {
        let info = MkvVideoInfo {
            pixel_width: 1920,
            pixel_height: 1080,
            display_width: None,
            display_height: None,
        };
        assert_eq!(info.pixel_width, 1920);
        assert_eq!(info.pixel_height, 1080);
    }

    #[test]
    fn test_mkv_audio_info() {
        let info = MkvAudioInfo {
            sampling_frequency: 48000.0,
            output_sampling_frequency: None,
            channels: 2,
            bit_depth: Some(16),
        };
        assert!((info.sampling_frequency - 48000.0).abs() < f64::EPSILON);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bit_depth, Some(16));
    }

    #[test]
    fn test_mkv_track_info() {
        let track = MkvTrackInfo {
            track_number: 1,
            track_type: MkvTrackType::Video,
            codec_id: "V_MPEG4/ISO/AVC".to_string(),
            codec_private: Some(vec![0x01, 0x64, 0x00, 0x1F]),
            default_duration_ns: Some(33_333_333),
            video: Some(MkvVideoInfo {
                pixel_width: 1920,
                pixel_height: 1080,
                display_width: None,
                display_height: None,
            }),
            audio: None,
        };
        assert_eq!(track.track_number, 1);
        assert_eq!(track.track_type, MkvTrackType::Video);
        assert!(track.video.is_some());
        assert!(track.audio.is_none());
    }

    #[test]
    fn test_mkv_cue_point() {
        let cue = MkvCuePoint {
            time: 5000,
            track: 1,
            cluster_position: 12345,
            relative_position: Some(0),
        };
        assert_eq!(cue.time, 5000);
        assert_eq!(cue.cluster_position, 12345);
    }
}
