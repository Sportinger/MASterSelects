//! MP4 box (atom) writers for ISO Base Media File Format (ISO 14496-12).
//!
//! This module writes the structural boxes that make up an MP4 file:
//! ftyp, moov (mvhd, trak, tkhd, mdia, mdhd, hdlr, minf, stbl, etc.)
//!
//! The mdat (media data) box is written progressively by the muxer.

use byteorder::{BigEndian, WriteBytesExt};
use ms_common::{AudioCodec, VideoCodec};
use std::io::{Seek, Write};

use crate::atoms::{
    self, box_size_placeholder, encode_language, fill_box_size, mp4_creation_time,
    seconds_to_timescale, write_box_header, write_fixed_point_16_16, write_fixed_point_8_8,
    write_full_box_header, write_zeros, MOVIE_TIMESCALE,
};
use crate::error::MuxResult;

/// Information about a single sample (frame) in a track.
#[derive(Clone, Debug)]
pub struct SampleInfo {
    /// Byte offset of sample in the file (within mdat).
    pub offset: u64,
    /// Size of sample in bytes.
    pub size: u32,
    /// Duration of this sample in timescale units.
    pub duration: u32,
    /// Composition time offset (PTS - DTS) in timescale units.
    pub composition_offset: i32,
    /// Whether this is a sync sample (keyframe).
    pub is_sync: bool,
}

/// Describes a track to be written into the moov box.
#[derive(Clone, Debug)]
pub struct TrackInfo {
    /// 1-based track ID.
    pub track_id: u32,
    /// Track timescale.
    pub timescale: u32,
    /// Total duration in timescale units.
    pub duration: u64,
    /// Handler type: video or audio.
    pub handler: TrackHandler,
    /// All samples in this track.
    pub samples: Vec<SampleInfo>,
}

/// Track media handler type.
#[derive(Clone, Debug)]
pub enum TrackHandler {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
        /// SPS NAL unit (H.264) or VPS+SPS+PPS (H.265).
        sps: Vec<u8>,
        /// PPS NAL unit.
        pps: Vec<u8>,
    },
    Audio {
        codec: AudioCodec,
        sample_rate: u32,
        channels: u16,
        /// Codec-specific config (e.g. AudioSpecificConfig for AAC).
        config_data: Vec<u8>,
    },
}

/// Write the ftyp (File Type) box.
///
/// Compatible brands: isom, iso6, mp41
pub fn write_ftyp<W: Write>(writer: &mut W) -> MuxResult<()> {
    // ftyp box:
    //   major_brand: isom
    //   minor_version: 0x200
    //   compatible_brands: isom, iso6, mp41
    let size: u32 = 8 + 4 + 4 + 4 * 3; // header + major + minor + 3 brands = 28
    write_box_header(writer, b"ftyp", size)?;
    writer.write_all(b"isom")?; // major brand
    writer.write_u32::<BigEndian>(0x200)?; // minor version
    writer.write_all(b"isom")?; // compatible brand 1
    writer.write_all(b"iso6")?; // compatible brand 2
    writer.write_all(b"mp41")?; // compatible brand 3
    Ok(())
}

/// Write the mvhd (Movie Header) box — version 0.
///
/// `duration_secs` is the movie duration in seconds.
pub fn write_mvhd<W: Write + Seek>(writer: &mut W, duration_secs: f64) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"mvhd")?;

    let creation_time = mp4_creation_time();
    let duration = seconds_to_timescale(duration_secs, MOVIE_TIMESCALE);

    // version=0, flags=0
    writer.write_u32::<BigEndian>(0)?; // version + flags

    writer.write_u32::<BigEndian>(creation_time as u32)?; // creation_time
    writer.write_u32::<BigEndian>(creation_time as u32)?; // modification_time
    writer.write_u32::<BigEndian>(MOVIE_TIMESCALE)?; // timescale
    writer.write_u32::<BigEndian>(duration as u32)?; // duration

    write_fixed_point_16_16(writer, 1.0)?; // rate (1.0 = normal)
    write_fixed_point_8_8(writer, 1.0)?; // volume (1.0 = full)

    write_zeros(writer, 10)?; // reserved

    // Unity matrix (3x3 identity in 16.16 fixed point, except [2][2] is 30.2)
    // Row 1
    write_fixed_point_16_16(writer, 1.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    // Row 2
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 1.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    // Row 3
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    writer.write_u32::<BigEndian>(0x4000_0000)?; // 1.0 in 30.2 fixed point

    write_zeros(writer, 24)?; // pre-defined (6 x u32)

    writer.write_u32::<BigEndian>(0xFFFF_FFFF)?; // next_track_ID (use max)

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the tkhd (Track Header) box — version 0.
pub fn write_tkhd<W: Write + Seek>(
    writer: &mut W,
    track_id: u32,
    duration_secs: f64,
    width: u32,
    height: u32,
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"tkhd")?;

    let creation_time = mp4_creation_time();
    let duration = seconds_to_timescale(duration_secs, MOVIE_TIMESCALE);

    // version=0, flags=0x000003 (track_enabled | track_in_movie)
    writer.write_u32::<BigEndian>(0x00_000003)?;

    writer.write_u32::<BigEndian>(creation_time as u32)?; // creation_time
    writer.write_u32::<BigEndian>(creation_time as u32)?; // modification_time
    writer.write_u32::<BigEndian>(track_id)?; // track_ID
    write_zeros(writer, 4)?; // reserved
    writer.write_u32::<BigEndian>(duration as u32)?; // duration

    write_zeros(writer, 8)?; // reserved (2 x u32)
    writer.write_i16::<BigEndian>(0)?; // layer
    writer.write_i16::<BigEndian>(0)?; // alternate_group
    // Volume: 0x0100 for audio, 0 for video
    if width == 0 && height == 0 {
        write_fixed_point_8_8(writer, 1.0)?; // audio track
    } else {
        write_fixed_point_8_8(writer, 0.0)?; // video track
    }
    write_zeros(writer, 2)?; // reserved

    // Unity matrix
    write_fixed_point_16_16(writer, 1.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 1.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    write_fixed_point_16_16(writer, 0.0)?;
    writer.write_u32::<BigEndian>(0x4000_0000)?;

    // Width and height in 16.16 fixed point
    write_fixed_point_16_16(writer, width as f64)?;
    write_fixed_point_16_16(writer, height as f64)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the mdhd (Media Header) box — version 0.
pub fn write_mdhd<W: Write + Seek>(
    writer: &mut W,
    timescale: u32,
    duration: u64,
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"mdhd")?;

    let creation_time = mp4_creation_time();

    // version=0, flags=0
    writer.write_u32::<BigEndian>(0)?;

    writer.write_u32::<BigEndian>(creation_time as u32)?;
    writer.write_u32::<BigEndian>(creation_time as u32)?;
    writer.write_u32::<BigEndian>(timescale)?;
    writer.write_u32::<BigEndian>(duration as u32)?;

    // Language: "und" (undetermined)
    writer.write_u16::<BigEndian>(encode_language("und"))?;
    // Pre-defined
    writer.write_u16::<BigEndian>(0)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the hdlr (Handler Reference) box.
///
/// `handler_type` should be "vide" for video or "soun" for audio.
pub fn write_hdlr<W: Write + Seek>(
    writer: &mut W,
    handler_type: &[u8; 4],
) -> MuxResult<()> {
    let name = match handler_type {
        b"vide" => "VideoHandler\0",
        b"soun" => "SoundHandler\0",
        _ => "DataHandler\0",
    };

    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"hdlr")?;

    // version=0, flags=0
    writer.write_u32::<BigEndian>(0)?;

    write_zeros(writer, 4)?; // pre_defined
    writer.write_all(handler_type)?; // handler_type
    write_zeros(writer, 12)?; // reserved (3 x u32)
    writer.write_all(name.as_bytes())?; // name (null-terminated)

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the avcC (AVC Decoder Configuration Record) box.
///
/// This contains the SPS and PPS NAL units needed to initialize an H.264 decoder.
fn write_avcc<W: Write + Seek>(
    writer: &mut W,
    sps: &[u8],
    pps: &[u8],
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"avcC")?;

    // AVCDecoderConfigurationRecord
    writer.write_u8(1)?; // configurationVersion
    writer.write_u8(if sps.len() > 1 { sps[1] } else { 0x64 })?; // AVCProfileIndication
    writer.write_u8(if sps.len() > 2 { sps[2] } else { 0x00 })?; // profile_compatibility
    writer.write_u8(if sps.len() > 3 { sps[3] } else { 0x1F })?; // AVCLevelIndication
    writer.write_u8(0xFF)?; // lengthSizeMinusOne = 3 (4-byte NAL length) | reserved bits

    // SPS
    writer.write_u8(0xE1)?; // numOfSequenceParameterSets = 1 | reserved bits
    writer.write_u16::<BigEndian>(sps.len() as u16)?;
    writer.write_all(sps)?;

    // PPS
    writer.write_u8(1)?; // numOfPictureParameterSets
    writer.write_u16::<BigEndian>(pps.len() as u16)?;
    writer.write_all(pps)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the hvcC (HEVC Decoder Configuration Record) box.
fn write_hvcc<W: Write + Seek>(
    writer: &mut W,
    sps: &[u8],
    pps: &[u8],
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"hvcC")?;

    // Simplified HEVCDecoderConfigurationRecord
    writer.write_u8(1)?; // configurationVersion

    // General profile/tier/level info — use defaults
    // general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5)
    writer.write_u8(0x01)?; // Main profile
    writer.write_u32::<BigEndian>(0x6000_0000)?; // general_profile_compatibility_flags
    // general_constraint_indicator_flags (6 bytes)
    write_zeros(writer, 6)?;
    writer.write_u8(93)?; // general_level_idc (level 3.1 = 93)

    writer.write_u16::<BigEndian>(0xF000)?; // min_spatial_segmentation_idc with reserved bits
    writer.write_u8(0xFC)?; // parallelismType with reserved bits
    writer.write_u8(0xFD)?; // chromaFormat with reserved bits (1 = 4:2:0)
    writer.write_u8(0xF8)?; // bitDepthLumaMinus8 with reserved bits
    writer.write_u8(0xF8)?; // bitDepthChromaMinus8 with reserved bits
    writer.write_u16::<BigEndian>(0)?; // avgFrameRate
    writer.write_u8(0x0F)?; // constantFrameRate(2) + numTemporalLayers(3) + temporalIdNested(1) + lengthSizeMinusOne(2) = 0x0F

    // numOfArrays
    writer.write_u8(2)?;

    // VPS/SPS array (NAL type 33 = SPS for HEVC)
    writer.write_u8(0x21)?; // array_completeness(1) + reserved(1) + NAL_unit_type(6) = SPS
    writer.write_u16::<BigEndian>(1)?; // numNalus
    writer.write_u16::<BigEndian>(sps.len() as u16)?;
    writer.write_all(sps)?;

    // PPS array (NAL type 34 = PPS for HEVC)
    writer.write_u8(0x22)?;
    writer.write_u16::<BigEndian>(1)?;
    writer.write_u16::<BigEndian>(pps.len() as u16)?;
    writer.write_all(pps)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the stsd (Sample Description) box for a video track.
pub fn write_stsd_video<W: Write + Seek>(
    writer: &mut W,
    codec: VideoCodec,
    width: u32,
    height: u32,
    sps: &[u8],
    pps: &[u8],
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stsd")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(1)?; // entry_count

    // Write the sample entry (avc1 or hvc1)
    let entry_size_pos = box_size_placeholder(writer)?;
    match codec {
        VideoCodec::H264 => writer.write_all(b"avc1")?,
        VideoCodec::H265 => writer.write_all(b"hvc1")?,
        _ => {
            return Err(crate::error::MuxError::InvalidConfig(format!(
                "Unsupported video codec for MP4 stsd: {:?}",
                codec
            )));
        }
    }

    // VisualSampleEntry fields
    write_zeros(writer, 6)?; // reserved
    writer.write_u16::<BigEndian>(1)?; // data_reference_index
    write_zeros(writer, 2)?; // pre_defined
    write_zeros(writer, 2)?; // reserved
    write_zeros(writer, 12)?; // pre_defined (3 x u32)
    writer.write_u16::<BigEndian>(width as u16)?;
    writer.write_u16::<BigEndian>(height as u16)?;
    writer.write_u32::<BigEndian>(0x0048_0000)?; // horizresolution (72 dpi, 16.16)
    writer.write_u32::<BigEndian>(0x0048_0000)?; // vertresolution (72 dpi, 16.16)
    write_zeros(writer, 4)?; // reserved
    writer.write_u16::<BigEndian>(1)?; // frame_count
    write_zeros(writer, 32)?; // compressorname (32 bytes, empty)
    writer.write_u16::<BigEndian>(0x0018)?; // depth (24-bit color)
    writer.write_i16::<BigEndian>(-1)?; // pre_defined

    // Write codec-specific config box
    match codec {
        VideoCodec::H264 => write_avcc(writer, sps, pps)?,
        VideoCodec::H265 => write_hvcc(writer, sps, pps)?,
        _ => unreachable!(),
    }

    fill_box_size(writer, entry_size_pos)?;
    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the stsd (Sample Description) box for an audio track.
pub fn write_stsd_audio<W: Write + Seek>(
    writer: &mut W,
    codec: AudioCodec,
    sample_rate: u32,
    channels: u16,
    config_data: &[u8],
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stsd")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(1)?; // entry_count

    let entry_size_pos = box_size_placeholder(writer)?;
    match codec {
        AudioCodec::Aac => writer.write_all(b"mp4a")?,
        AudioCodec::Opus => writer.write_all(b"Opus")?,
        _ => {
            return Err(crate::error::MuxError::InvalidConfig(format!(
                "Unsupported audio codec for MP4 stsd: {:?}",
                codec
            )));
        }
    }

    // AudioSampleEntry fields
    write_zeros(writer, 6)?; // reserved
    writer.write_u16::<BigEndian>(1)?; // data_reference_index
    write_zeros(writer, 8)?; // reserved (2 x u32)
    writer.write_u16::<BigEndian>(channels)?; // channelcount
    writer.write_u16::<BigEndian>(16)?; // samplesize (16-bit)
    write_zeros(writer, 2)?; // pre_defined
    write_zeros(writer, 2)?; // reserved
    // Sample rate in 16.16 fixed point
    writer.write_u32::<BigEndian>(sample_rate << 16)?;

    // Codec-specific boxes
    match codec {
        AudioCodec::Aac => {
            write_esds(writer, config_data, sample_rate, channels)?;
        }
        AudioCodec::Opus => {
            write_dops(writer, sample_rate, channels)?;
        }
        _ => unreachable!(),
    }

    fill_box_size(writer, entry_size_pos)?;
    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the esds (Elementary Stream Descriptor) box for AAC.
fn write_esds<W: Write + Seek>(
    writer: &mut W,
    config_data: &[u8],
    _sample_rate: u32,
    _channels: u16,
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"esds")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags

    // ES_Descriptor tag=0x03
    let dec_config_len = 13 + 5 + config_data.len();
    let es_desc_len = 3 + 5 + dec_config_len;

    writer.write_u8(0x03)?; // ES_DescrTag
    write_descr_length(writer, es_desc_len)?;
    writer.write_u16::<BigEndian>(1)?; // ES_ID
    writer.write_u8(0)?; // stream priority

    // DecoderConfigDescriptor tag=0x04
    writer.write_u8(0x04)?; // DecoderConfigDescrTag
    write_descr_length(writer, dec_config_len)?;
    writer.write_u8(0x40)?; // objectTypeIndication = Audio ISO/IEC 14496-3 (AAC)
    writer.write_u8(0x15)?; // streamType = Audio stream
    write_zeros(writer, 3)?; // bufferSizeDB (24-bit)
    writer.write_u32::<BigEndian>(128_000)?; // maxBitrate
    writer.write_u32::<BigEndian>(128_000)?; // avgBitrate

    // DecoderSpecificInfo tag=0x05
    writer.write_u8(0x05)?; // DecoderSpecificInfoTag
    write_descr_length(writer, config_data.len())?;
    writer.write_all(config_data)?;

    // SLConfigDescriptor tag=0x06
    writer.write_u8(0x06)?; // SLConfigDescrTag
    write_descr_length(writer, 1)?;
    writer.write_u8(0x02)?; // predefined = MP4

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write MPEG-4 descriptor length in expandable form (1-4 bytes).
fn write_descr_length<W: Write>(writer: &mut W, len: usize) -> MuxResult<()> {
    // Simple form: for lengths < 128, just write the byte
    if len < 128 {
        writer.write_u8(len as u8)?;
    } else {
        // Expandable size encoding (up to 4 bytes)
        let mut val = len;
        let mut bytes = Vec::new();
        loop {
            bytes.push((val & 0x7F) as u8);
            val >>= 7;
            if val == 0 {
                break;
            }
        }
        bytes.reverse();
        for (i, b) in bytes.iter().enumerate() {
            if i < bytes.len() - 1 {
                writer.write_u8(b | 0x80)?;
            } else {
                writer.write_u8(*b)?;
            }
        }
    }
    Ok(())
}

/// Write the dOps (Opus Specific) box.
fn write_dops<W: Write + Seek>(
    writer: &mut W,
    sample_rate: u32,
    channels: u16,
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"dOps")?;

    writer.write_u8(0)?; // version
    writer.write_u8(channels as u8)?; // OutputChannelCount
    writer.write_u16::<BigEndian>(312)?; // PreSkip (standard value)
    writer.write_u32::<BigEndian>(sample_rate)?; // InputSampleRate
    writer.write_i16::<BigEndian>(0)?; // OutputGain
    writer.write_u8(0)?; // ChannelMappingFamily

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the stbl (Sample Table) box containing all sample metadata.
pub fn write_stbl<W: Write + Seek>(
    writer: &mut W,
    samples: &[SampleInfo],
    handler: &TrackHandler,
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stbl")?;

    // stsd (Sample Description)
    match handler {
        TrackHandler::Video {
            codec,
            width,
            height,
            sps,
            pps,
        } => {
            write_stsd_video(writer, *codec, *width, *height, sps, pps)?;
        }
        TrackHandler::Audio {
            codec,
            sample_rate,
            channels,
            config_data,
        } => {
            write_stsd_audio(writer, *codec, *sample_rate, *channels, config_data)?;
        }
    }

    // stts (Decoding Time to Sample)
    write_stts(writer, samples)?;

    // ctts (Composition Time to Sample) — only if any composition offsets are nonzero
    let has_ctts = samples.iter().any(|s| s.composition_offset != 0);
    if has_ctts {
        write_ctts(writer, samples)?;
    }

    // stsc (Sample to Chunk)
    write_stsc(writer, samples)?;

    // stsz (Sample Size)
    write_stsz(writer, samples)?;

    // stco or co64 (Chunk Offset)
    let needs_co64 = samples.iter().any(|s| s.offset > u32::MAX as u64);
    if needs_co64 {
        write_co64(writer, samples)?;
    } else {
        write_stco(writer, samples)?;
    }

    // stss (Sync Sample) — only for video tracks
    if matches!(handler, TrackHandler::Video { .. }) {
        write_stss(writer, samples)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write stts (Decoding Time to Sample) box — run-length encoded durations.
fn write_stts<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    // Run-length encode the durations
    let entries = run_length_encode_durations(samples);

    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stts")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(entries.len() as u32)?;

    for (count, duration) in &entries {
        writer.write_u32::<BigEndian>(*count)?;
        writer.write_u32::<BigEndian>(*duration)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Run-length encode sample durations: Vec<(count, duration)>.
fn run_length_encode_durations(samples: &[SampleInfo]) -> Vec<(u32, u32)> {
    if samples.is_empty() {
        return vec![];
    }
    let mut entries = Vec::new();
    let mut current_duration = samples[0].duration;
    let mut count = 1u32;

    for sample in &samples[1..] {
        if sample.duration == current_duration {
            count += 1;
        } else {
            entries.push((count, current_duration));
            current_duration = sample.duration;
            count = 1;
        }
    }
    entries.push((count, current_duration));
    entries
}

/// Write ctts (Composition Time to Sample) box — version 1 (signed offsets).
fn write_ctts<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    // Run-length encode composition offsets
    let entries = run_length_encode_ctts(samples);

    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"ctts")?;
    // version=1 for signed composition offsets
    writer.write_u32::<BigEndian>(0x0100_0000)?;
    writer.write_u32::<BigEndian>(entries.len() as u32)?;

    for (count, offset) in &entries {
        writer.write_u32::<BigEndian>(*count)?;
        writer.write_i32::<BigEndian>(*offset)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Run-length encode composition offsets: Vec<(count, offset)>.
fn run_length_encode_ctts(samples: &[SampleInfo]) -> Vec<(u32, i32)> {
    if samples.is_empty() {
        return vec![];
    }
    let mut entries = Vec::new();
    let mut current_offset = samples[0].composition_offset;
    let mut count = 1u32;

    for sample in &samples[1..] {
        if sample.composition_offset == current_offset {
            count += 1;
        } else {
            entries.push((count, current_offset));
            current_offset = sample.composition_offset;
            count = 1;
        }
    }
    entries.push((count, current_offset));
    entries
}

/// Write stsc (Sample to Chunk) box.
///
/// We use the simplest strategy: one sample per chunk.
fn write_stsc<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stsc")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags

    if samples.is_empty() {
        writer.write_u32::<BigEndian>(0)?; // entry_count
    } else {
        // One entry: all chunks have 1 sample each, referencing sample description index 1
        writer.write_u32::<BigEndian>(1)?; // entry_count
        writer.write_u32::<BigEndian>(1)?; // first_chunk
        writer.write_u32::<BigEndian>(1)?; // samples_per_chunk
        writer.write_u32::<BigEndian>(1)?; // sample_description_index
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write stsz (Sample Size) box.
fn write_stsz<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stsz")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags

    // Check if all samples have the same size
    let all_same = if samples.is_empty() {
        true
    } else {
        samples.iter().all(|s| s.size == samples[0].size)
    };

    if all_same && !samples.is_empty() {
        writer.write_u32::<BigEndian>(samples[0].size)?; // sample_size (uniform)
        writer.write_u32::<BigEndian>(samples.len() as u32)?; // sample_count
    } else {
        writer.write_u32::<BigEndian>(0)?; // sample_size = 0 (variable)
        writer.write_u32::<BigEndian>(samples.len() as u32)?;
        for sample in samples {
            writer.write_u32::<BigEndian>(sample.size)?;
        }
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write stco (Chunk Offset) box — 32-bit offsets.
fn write_stco<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stco")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(samples.len() as u32)?;

    for sample in samples {
        writer.write_u32::<BigEndian>(sample.offset as u32)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write co64 (Chunk Offset 64-bit) box — for files > 4GB.
fn write_co64<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"co64")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(samples.len() as u32)?;

    for sample in samples {
        writer.write_u64::<BigEndian>(sample.offset)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write stss (Sync Sample) box — lists keyframe sample numbers (1-based).
fn write_stss<W: Write + Seek>(writer: &mut W, samples: &[SampleInfo]) -> MuxResult<()> {
    let sync_samples: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_sync)
        .map(|(i, _)| (i + 1) as u32) // 1-based
        .collect();

    // If all samples are sync, we can skip stss (or write it anyway for clarity)
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"stss")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(sync_samples.len() as u32)?;

    for sample_number in &sync_samples {
        writer.write_u32::<BigEndian>(*sample_number)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the dinf (Data Information) box with a dref (Data Reference) sub-box.
fn write_dinf<W: Write + Seek>(writer: &mut W) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"dinf")?;

    // dref box
    let dref_size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"dref")?;
    writer.write_u32::<BigEndian>(0)?; // version + flags
    writer.write_u32::<BigEndian>(1)?; // entry_count

    // url entry (self-contained: data is in same file)
    write_full_box_header(writer, b"url ", 12, 0, 0x000001)?; // flag 1 = self-contained

    fill_box_size(writer, dref_size_pos)?;
    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the minf (Media Information) box.
fn write_minf<W: Write + Seek>(
    writer: &mut W,
    handler: &TrackHandler,
    samples: &[SampleInfo],
) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"minf")?;

    // Media-specific header
    match handler {
        TrackHandler::Video { .. } => {
            // vmhd (Video Media Header)
            write_full_box_header(writer, b"vmhd", 20, 0, 0x000001)?;
            writer.write_u16::<BigEndian>(0)?; // graphicsmode
            write_zeros(writer, 6)?; // opcolor (3 x u16)
        }
        TrackHandler::Audio { .. } => {
            // smhd (Sound Media Header)
            write_full_box_header(writer, b"smhd", 16, 0, 0)?;
            writer.write_i16::<BigEndian>(0)?; // balance
            write_zeros(writer, 2)?; // reserved
        }
    }

    // dinf (Data Information)
    write_dinf(writer)?;

    // stbl (Sample Table)
    write_stbl(writer, samples, handler)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the mdia (Media) box for a track.
fn write_mdia<W: Write + Seek>(writer: &mut W, track: &TrackInfo) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"mdia")?;

    // mdhd
    write_mdhd(writer, track.timescale, track.duration)?;

    // hdlr
    let handler_type = match &track.handler {
        TrackHandler::Video { .. } => b"vide",
        TrackHandler::Audio { .. } => b"soun",
    };
    write_hdlr(writer, handler_type)?;

    // minf
    write_minf(writer, &track.handler, &track.samples)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write a complete trak (Track) box.
fn write_trak<W: Write + Seek>(writer: &mut W, track: &TrackInfo) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"trak")?;

    // Duration in seconds for tkhd (uses movie timescale)
    let duration_secs = atoms::timescale_to_seconds(track.duration, track.timescale);

    // tkhd
    let (width, height) = match &track.handler {
        TrackHandler::Video { width, height, .. } => (*width, *height),
        TrackHandler::Audio { .. } => (0, 0),
    };
    write_tkhd(writer, track.track_id, duration_secs, width, height)?;

    // mdia
    write_mdia(writer, track)?;

    fill_box_size(writer, size_pos)?;
    Ok(())
}

/// Write the complete moov (Movie) box with all tracks.
pub fn write_moov<W: Write + Seek>(writer: &mut W, tracks: &[TrackInfo]) -> MuxResult<()> {
    let size_pos = box_size_placeholder(writer)?;
    writer.write_all(b"moov")?;

    // Determine overall duration (max of all tracks)
    let max_duration_secs = tracks
        .iter()
        .map(|t| atoms::timescale_to_seconds(t.duration, t.timescale))
        .fold(0.0f64, f64::max);

    // mvhd
    write_mvhd(writer, max_duration_secs)?;

    // trak for each track
    for track in tracks {
        write_trak(writer, track)?;
    }

    fill_box_size(writer, size_pos)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper to extract a box type from a buffer at a given offset.
    fn box_type_at(buf: &[u8], offset: usize) -> &[u8] {
        &buf[offset + 4..offset + 8]
    }

    /// Helper to extract a box size from a buffer at a given offset.
    fn box_size_at(buf: &[u8], offset: usize) -> u32 {
        u32::from_be_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn test_write_ftyp() {
        let mut buf = Vec::new();
        write_ftyp(&mut buf).unwrap();
        assert_eq!(buf.len(), 28);
        assert_eq!(box_size_at(&buf, 0), 28);
        assert_eq!(box_type_at(&buf, 0), b"ftyp");
        // Major brand
        assert_eq!(&buf[8..12], b"isom");
        // Minor version
        assert_eq!(&buf[12..16], &[0x00, 0x00, 0x02, 0x00]);
        // Compatible brands
        assert_eq!(&buf[16..20], b"isom");
        assert_eq!(&buf[20..24], b"iso6");
        assert_eq!(&buf[24..28], b"mp41");
    }

    #[test]
    fn test_write_mvhd() {
        let mut cursor = Cursor::new(Vec::new());
        write_mvhd(&mut cursor, 10.0).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"mvhd");
        let size = box_size_at(&buf, 0);
        assert_eq!(buf.len(), size as usize);
    }

    #[test]
    fn test_write_tkhd_video() {
        let mut cursor = Cursor::new(Vec::new());
        write_tkhd(&mut cursor, 1, 5.0, 1920, 1080).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"tkhd");
        let size = box_size_at(&buf, 0);
        assert_eq!(buf.len(), size as usize);
    }

    #[test]
    fn test_write_tkhd_audio() {
        let mut cursor = Cursor::new(Vec::new());
        write_tkhd(&mut cursor, 2, 5.0, 0, 0).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"tkhd");
    }

    #[test]
    fn test_write_mdhd() {
        let mut cursor = Cursor::new(Vec::new());
        write_mdhd(&mut cursor, 90_000, 900_000).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"mdhd");
        let size = box_size_at(&buf, 0);
        assert_eq!(buf.len(), size as usize);
    }

    #[test]
    fn test_write_hdlr_video() {
        let mut cursor = Cursor::new(Vec::new());
        write_hdlr(&mut cursor, b"vide").unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"hdlr");
        // Handler type should be at offset 16 (8 header + 4 version+flags + 4 pre_defined)
        assert_eq!(&buf[16..20], b"vide");
    }

    #[test]
    fn test_write_hdlr_audio() {
        let mut cursor = Cursor::new(Vec::new());
        write_hdlr(&mut cursor, b"soun").unwrap();
        let buf = cursor.into_inner();
        assert_eq!(&buf[16..20], b"soun");
    }

    #[test]
    fn test_write_stsd_video_h264() {
        let sps = vec![0x67, 0x64, 0x00, 0x1F, 0xAC, 0xD9, 0x40];
        let pps = vec![0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0];
        let mut cursor = Cursor::new(Vec::new());
        write_stsd_video(&mut cursor, VideoCodec::H264, 1920, 1080, &sps, &pps).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"stsd");
        // Should contain avc1 sample entry
        assert!(buf.windows(4).any(|w| w == b"avc1"));
        // Should contain avcC box
        assert!(buf.windows(4).any(|w| w == b"avcC"));
    }

    #[test]
    fn test_write_stsd_video_h265() {
        let sps = vec![0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00];
        let pps = vec![0x44, 0x01, 0xC1, 0x73, 0xD1, 0x89];
        let mut cursor = Cursor::new(Vec::new());
        write_stsd_video(&mut cursor, VideoCodec::H265, 3840, 2160, &sps, &pps).unwrap();
        let buf = cursor.into_inner();
        assert!(buf.windows(4).any(|w| w == b"hvc1"));
        assert!(buf.windows(4).any(|w| w == b"hvcC"));
    }

    #[test]
    fn test_write_stsd_video_unsupported() {
        let mut cursor = Cursor::new(Vec::new());
        let result = write_stsd_video(&mut cursor, VideoCodec::Vp9, 1920, 1080, &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_stsd_audio_aac() {
        let config = vec![0x12, 0x10]; // AAC-LC, 44100 Hz, stereo
        let mut cursor = Cursor::new(Vec::new());
        write_stsd_audio(&mut cursor, AudioCodec::Aac, 44100, 2, &config).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"stsd");
        assert!(buf.windows(4).any(|w| w == b"mp4a"));
        assert!(buf.windows(4).any(|w| w == b"esds"));
    }

    #[test]
    fn test_write_stsd_audio_opus() {
        let mut cursor = Cursor::new(Vec::new());
        write_stsd_audio(&mut cursor, AudioCodec::Opus, 48000, 2, &[]).unwrap();
        let buf = cursor.into_inner();
        assert!(buf.windows(4).any(|w| w == b"Opus"));
        assert!(buf.windows(4).any(|w| w == b"dOps"));
    }

    #[test]
    fn test_write_stsd_audio_unsupported() {
        let mut cursor = Cursor::new(Vec::new());
        let result = write_stsd_audio(&mut cursor, AudioCodec::Mp3, 44100, 2, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_stbl_with_samples() {
        let samples = vec![
            SampleInfo {
                offset: 100,
                size: 5000,
                duration: 3000,
                composition_offset: 0,
                is_sync: true,
            },
            SampleInfo {
                offset: 5100,
                size: 3000,
                duration: 3000,
                composition_offset: 0,
                is_sync: false,
            },
            SampleInfo {
                offset: 8100,
                size: 4000,
                duration: 3000,
                composition_offset: 0,
                is_sync: false,
            },
        ];
        let handler = TrackHandler::Video {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            sps: vec![0x67, 0x64, 0x00, 0x1F],
            pps: vec![0x68, 0xEB],
        };

        let mut cursor = Cursor::new(Vec::new());
        write_stbl(&mut cursor, &samples, &handler).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"stbl");

        // Should contain all required sub-boxes
        assert!(buf.windows(4).any(|w| w == b"stsd"));
        assert!(buf.windows(4).any(|w| w == b"stts"));
        assert!(buf.windows(4).any(|w| w == b"stsc"));
        assert!(buf.windows(4).any(|w| w == b"stsz"));
        assert!(buf.windows(4).any(|w| w == b"stco"));
        assert!(buf.windows(4).any(|w| w == b"stss"));
    }

    #[test]
    fn test_write_stbl_empty() {
        let handler = TrackHandler::Video {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            sps: vec![0x67],
            pps: vec![0x68],
        };
        let mut cursor = Cursor::new(Vec::new());
        write_stbl(&mut cursor, &[], &handler).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"stbl");
    }

    #[test]
    fn test_co64_for_large_offsets() {
        let samples = vec![SampleInfo {
            offset: 5_000_000_000, // > 4GB
            size: 1000,
            duration: 3000,
            composition_offset: 0,
            is_sync: true,
        }];
        let handler = TrackHandler::Video {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            sps: vec![0x67],
            pps: vec![0x68],
        };
        let mut cursor = Cursor::new(Vec::new());
        write_stbl(&mut cursor, &samples, &handler).unwrap();
        let buf = cursor.into_inner();
        // Should use co64 instead of stco
        assert!(buf.windows(4).any(|w| w == b"co64"));
        assert!(!buf.windows(4).any(|w| w == b"stco"));
    }

    #[test]
    fn test_ctts_written_when_needed() {
        let samples = vec![
            SampleInfo {
                offset: 100,
                size: 1000,
                duration: 3000,
                composition_offset: 3000, // nonzero
                is_sync: true,
            },
            SampleInfo {
                offset: 1100,
                size: 1000,
                duration: 3000,
                composition_offset: 6000,
                is_sync: false,
            },
        ];
        let handler = TrackHandler::Video {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            sps: vec![0x67],
            pps: vec![0x68],
        };
        let mut cursor = Cursor::new(Vec::new());
        write_stbl(&mut cursor, &samples, &handler).unwrap();
        let buf = cursor.into_inner();
        assert!(buf.windows(4).any(|w| w == b"ctts"));
    }

    #[test]
    fn test_write_moov_single_video_track() {
        let tracks = vec![TrackInfo {
            track_id: 1,
            timescale: 90_000,
            duration: 450_000, // 5 seconds
            handler: TrackHandler::Video {
                codec: VideoCodec::H264,
                width: 1920,
                height: 1080,
                sps: vec![0x67, 0x64, 0x00, 0x1F],
                pps: vec![0x68, 0xEB],
            },
            samples: vec![
                SampleInfo {
                    offset: 100,
                    size: 50_000,
                    duration: 3000,
                    composition_offset: 0,
                    is_sync: true,
                },
                SampleInfo {
                    offset: 50_100,
                    size: 10_000,
                    duration: 3000,
                    composition_offset: 0,
                    is_sync: false,
                },
            ],
        }];

        let mut cursor = Cursor::new(Vec::new());
        write_moov(&mut cursor, &tracks).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"moov");
        assert!(buf.windows(4).any(|w| w == b"mvhd"));
        assert!(buf.windows(4).any(|w| w == b"trak"));
        assert!(buf.windows(4).any(|w| w == b"tkhd"));
        assert!(buf.windows(4).any(|w| w == b"mdia"));
    }

    #[test]
    fn test_write_moov_video_and_audio() {
        let tracks = vec![
            TrackInfo {
                track_id: 1,
                timescale: 90_000,
                duration: 450_000,
                handler: TrackHandler::Video {
                    codec: VideoCodec::H264,
                    width: 1920,
                    height: 1080,
                    sps: vec![0x67],
                    pps: vec![0x68],
                },
                samples: vec![SampleInfo {
                    offset: 100,
                    size: 5000,
                    duration: 3000,
                    composition_offset: 0,
                    is_sync: true,
                }],
            },
            TrackInfo {
                track_id: 2,
                timescale: 44100,
                duration: 220500,
                handler: TrackHandler::Audio {
                    codec: AudioCodec::Aac,
                    sample_rate: 44100,
                    channels: 2,
                    config_data: vec![0x12, 0x10],
                },
                samples: vec![SampleInfo {
                    offset: 5100,
                    size: 1024,
                    duration: 1024,
                    composition_offset: 0,
                    is_sync: true,
                }],
            },
        ];

        let mut cursor = Cursor::new(Vec::new());
        write_moov(&mut cursor, &tracks).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"moov");
        // Should have both video and audio handlers
        assert!(buf.windows(4).any(|w| w == b"vide"));
        assert!(buf.windows(4).any(|w| w == b"soun"));
    }

    #[test]
    fn test_run_length_encode_durations_uniform() {
        let samples: Vec<SampleInfo> = (0..100)
            .map(|i| SampleInfo {
                offset: i * 1000,
                size: 1000,
                duration: 3000,
                composition_offset: 0,
                is_sync: i == 0,
            })
            .collect();
        let entries = run_length_encode_durations(&samples);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (100, 3000));
    }

    #[test]
    fn test_run_length_encode_durations_varied() {
        let samples = vec![
            SampleInfo {
                offset: 0,
                size: 100,
                duration: 3000,
                composition_offset: 0,
                is_sync: true,
            },
            SampleInfo {
                offset: 100,
                size: 100,
                duration: 3000,
                composition_offset: 0,
                is_sync: false,
            },
            SampleInfo {
                offset: 200,
                size: 100,
                duration: 6000,
                composition_offset: 0,
                is_sync: false,
            },
        ];
        let entries = run_length_encode_durations(&samples);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (2, 3000));
        assert_eq!(entries[1], (1, 6000));
    }

    #[test]
    fn test_run_length_encode_durations_empty() {
        let entries = run_length_encode_durations(&[]);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_stss_only_keyframes() {
        let samples = vec![
            SampleInfo {
                offset: 0,
                size: 5000,
                duration: 3000,
                composition_offset: 0,
                is_sync: true,
            },
            SampleInfo {
                offset: 5000,
                size: 1000,
                duration: 3000,
                composition_offset: 0,
                is_sync: false,
            },
            SampleInfo {
                offset: 6000,
                size: 1000,
                duration: 3000,
                composition_offset: 0,
                is_sync: false,
            },
            SampleInfo {
                offset: 7000,
                size: 5000,
                duration: 3000,
                composition_offset: 0,
                is_sync: true,
            },
        ];

        let mut cursor = Cursor::new(Vec::new());
        write_stss(&mut cursor, &samples).unwrap();
        let buf = cursor.into_inner();
        assert_eq!(box_type_at(&buf, 0), b"stss");
        // entry_count should be 2 (samples 1 and 4 are keyframes)
        let entry_count = u32::from_be_bytes(buf[12..16].try_into().unwrap());
        assert_eq!(entry_count, 2);
        // First keyframe = sample 1
        let kf1 = u32::from_be_bytes(buf[16..20].try_into().unwrap());
        assert_eq!(kf1, 1);
        // Second keyframe = sample 4
        let kf2 = u32::from_be_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(kf2, 4);
    }

    #[test]
    fn test_stsz_uniform_sizes() {
        let samples = vec![
            SampleInfo {
                offset: 0,
                size: 1024,
                duration: 1024,
                composition_offset: 0,
                is_sync: true,
            },
            SampleInfo {
                offset: 1024,
                size: 1024,
                duration: 1024,
                composition_offset: 0,
                is_sync: true,
            },
        ];

        let mut cursor = Cursor::new(Vec::new());
        write_stsz(&mut cursor, &samples).unwrap();
        let buf = cursor.into_inner();
        // sample_size should be 1024 (uniform), not 0
        let sample_size = u32::from_be_bytes(buf[12..16].try_into().unwrap());
        assert_eq!(sample_size, 1024);
        let sample_count = u32::from_be_bytes(buf[16..20].try_into().unwrap());
        assert_eq!(sample_count, 2);
        // No per-sample entries follow when uniform
        let size = box_size_at(&buf, 0);
        assert_eq!(size as usize, 20);
    }

    #[test]
    fn test_stsz_variable_sizes() {
        let samples = vec![
            SampleInfo {
                offset: 0,
                size: 5000,
                duration: 3000,
                composition_offset: 0,
                is_sync: true,
            },
            SampleInfo {
                offset: 5000,
                size: 3000,
                duration: 3000,
                composition_offset: 0,
                is_sync: false,
            },
        ];

        let mut cursor = Cursor::new(Vec::new());
        write_stsz(&mut cursor, &samples).unwrap();
        let buf = cursor.into_inner();
        // sample_size should be 0 (variable)
        let sample_size = u32::from_be_bytes(buf[12..16].try_into().unwrap());
        assert_eq!(sample_size, 0);
        // Per-sample sizes follow
        let s1 = u32::from_be_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(s1, 5000);
        let s2 = u32::from_be_bytes(buf[24..28].try_into().unwrap());
        assert_eq!(s2, 3000);
    }
}
