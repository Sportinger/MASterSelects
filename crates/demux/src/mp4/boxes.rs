//! ISO BMFF box (atom) parser.
//!
//! Parses the box hierarchy of MP4/MOV/M4V files:
//! ftyp, moov, trak, mdia, minf, stbl, and sample table boxes.
//!
//! Reference: ISO 14496-12 (ISO Base Media File Format).

use byteorder::{BigEndian, ReadBytesExt};
use ms_common::DemuxError;
use std::io::{Read, Seek, SeekFrom};
use tracing::{debug, trace};

// ─── Box FourCC constants ────────────────────────────────────────────

/// Convert 4 ASCII bytes to a u32 FourCC code.
const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    ((a as u32) << 24) | ((b as u32) << 16) | ((c as u32) << 8) | (d as u32)
}

pub const FTYP: u32 = fourcc(b'f', b't', b'y', b'p');
pub const MOOV: u32 = fourcc(b'm', b'o', b'o', b'v');
pub const MVHD: u32 = fourcc(b'm', b'v', b'h', b'd');
pub const TRAK: u32 = fourcc(b't', b'r', b'a', b'k');
pub const TKHD: u32 = fourcc(b't', b'k', b'h', b'd');
pub const MDIA: u32 = fourcc(b'm', b'd', b'i', b'a');
pub const MDHD: u32 = fourcc(b'm', b'd', b'h', b'd');
pub const HDLR: u32 = fourcc(b'h', b'd', b'l', b'r');
pub const MINF: u32 = fourcc(b'm', b'i', b'n', b'f');
pub const STBL: u32 = fourcc(b's', b't', b'b', b'l');
pub const STSD: u32 = fourcc(b's', b't', b's', b'd');
pub const STTS: u32 = fourcc(b's', b't', b't', b's');
pub const STSC: u32 = fourcc(b's', b't', b's', b'c');
pub const STSZ: u32 = fourcc(b's', b't', b's', b'z');
pub const STCO: u32 = fourcc(b's', b't', b'c', b'o');
pub const CO64: u32 = fourcc(b'c', b'o', b'6', b'4');
pub const STSS: u32 = fourcc(b's', b't', b's', b's');
pub const CTTS: u32 = fourcc(b'c', b't', b't', b's');
pub const MDAT: u32 = fourcc(b'm', b'd', b'a', b't');
pub const AVCC: u32 = fourcc(b'a', b'v', b'c', b'C');
pub const AVC1: u32 = fourcc(b'a', b'v', b'c', b'1');
pub const AVC3: u32 = fourcc(b'a', b'v', b'c', b'3');
pub const HEV1: u32 = fourcc(b'h', b'e', b'v', b'1');
pub const HVC1: u32 = fourcc(b'h', b'v', b'c', b'1');
pub const HVCC: u32 = fourcc(b'h', b'v', b'c', b'C');
pub const VIDE: u32 = fourcc(b'v', b'i', b'd', b'e');
pub const SOUN: u32 = fourcc(b's', b'o', b'u', b'n');
pub const MP4A: u32 = fourcc(b'm', b'p', b'4', b'a');
pub const ESDS: u32 = fourcc(b'e', b's', b'd', b's');
pub const OPUS: u32 = fourcc(b'O', b'p', b'u', b's');
pub const DOPS: u32 = fourcc(b'd', b'O', b'p', b's');
pub const AC3_: u32 = fourcc(b'a', b'c', b'-', b'3');
pub const FLAC: u32 = fourcc(b'f', b'L', b'a', b'C');
pub const WAVE: u32 = fourcc(b'w', b'a', b'v', b'e');

/// Convert a FourCC u32 to a human-readable string for logging.
pub fn fourcc_to_string(cc: u32) -> String {
    let bytes = cc.to_be_bytes();
    bytes
        .iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '?'
            }
        })
        .collect()
}

// ─── Box Header ─────────────────────────────────────────────────────

/// A parsed ISO BMFF box header.
#[derive(Clone, Debug)]
pub struct BoxHeader {
    /// FourCC type code.
    pub box_type: u32,
    /// Total box size (including header). 0 means "extends to EOF".
    pub size: u64,
    /// Offset of the box start in the file.
    pub offset: u64,
    /// Size of the header itself (8 or 16 bytes).
    pub header_size: u8,
}

impl BoxHeader {
    /// Byte offset where the box content (payload) starts.
    pub fn content_offset(&self) -> u64 {
        self.offset + self.header_size as u64
    }

    /// Byte size of the content (payload), excluding the header.
    /// Returns None if the box extends to EOF (size == 0).
    pub fn content_size(&self) -> Option<u64> {
        if self.size == 0 {
            None
        } else {
            Some(self.size - self.header_size as u64)
        }
    }

    /// Byte offset of the first byte after this box.
    /// Returns None if the box extends to EOF.
    pub fn end_offset(&self) -> Option<u64> {
        if self.size == 0 {
            None
        } else {
            Some(self.offset + self.size)
        }
    }
}

/// Read a box header from the current position. Returns None at EOF.
pub fn read_box_header<R: Read + Seek>(reader: &mut R) -> Result<Option<BoxHeader>, DemuxError> {
    let offset = reader.stream_position().map_err(DemuxError::Io)?;

    // Try reading the first 4 bytes (size). If we get 0 bytes, we're at EOF.
    let size32 = match reader.read_u32::<BigEndian>() {
        Ok(v) => v,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(DemuxError::Io(e)),
    };

    let box_type = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;

    let (size, header_size) = match size32 {
        0 => {
            // Box extends to EOF
            (0u64, 8u8)
        }
        1 => {
            // 64-bit extended size follows
            let size64 = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
            (size64, 16u8)
        }
        _ => (size32 as u64, 8u8),
    };

    // Sanity check: size must be at least header_size (except for size==0 meaning EOF)
    if size != 0 && size < header_size as u64 {
        return Err(DemuxError::InvalidStructure {
            offset,
            reason: format!(
                "Box '{}' has invalid size {} (less than header)",
                fourcc_to_string(box_type),
                size
            ),
        });
    }

    trace!(
        "Box '{}' at offset {}, size {}",
        fourcc_to_string(box_type),
        offset,
        if size == 0 {
            "to-EOF".to_string()
        } else {
            size.to_string()
        }
    );

    Ok(Some(BoxHeader {
        box_type,
        size,
        offset,
        header_size,
    }))
}

/// Skip past the current box (seek to its end).
pub fn skip_box<R: Read + Seek>(reader: &mut R, header: &BoxHeader) -> Result<(), DemuxError> {
    match header.end_offset() {
        Some(end) => {
            reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
            Ok(())
        }
        None => {
            // Box extends to EOF — seek to end
            reader.seek(SeekFrom::End(0)).map_err(DemuxError::Io)?;
            Ok(())
        }
    }
}

// ─── ftyp Box ───────────────────────────────────────────────────────

/// Parsed ftyp (File Type) box.
#[derive(Clone, Debug)]
pub struct FtypBox {
    pub major_brand: u32,
    pub minor_version: u32,
    pub compatible_brands: Vec<u32>,
}

/// Parse an ftyp box. Reader must be positioned at the content start (after header).
pub fn parse_ftyp<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<FtypBox, DemuxError> {
    let content_size = header
        .content_size()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "ftyp box cannot extend to EOF".into(),
        })?;

    let major_brand = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let minor_version = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;

    let remaining = content_size.saturating_sub(8);
    let brand_count = remaining / 4;
    let mut compatible_brands = Vec::with_capacity(brand_count as usize);
    for _ in 0..brand_count {
        compatible_brands.push(reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?);
    }

    debug!(
        "ftyp: major_brand='{}', minor_version={}, {} compatible brands",
        fourcc_to_string(major_brand),
        minor_version,
        compatible_brands.len()
    );

    Ok(FtypBox {
        major_brand,
        minor_version,
        compatible_brands,
    })
}

// ─── mvhd Box ───────────────────────────────────────────────────────

/// Parsed mvhd (Movie Header) box — contains global timescale and duration.
#[derive(Clone, Debug)]
pub struct MvhdBox {
    pub timescale: u32,
    pub duration: u64,
}

/// Parse an mvhd box. Reader must be at content start.
pub fn parse_mvhd<R: Read + Seek>(reader: &mut R) -> Result<MvhdBox, DemuxError> {
    let version = reader.read_u8().map_err(DemuxError::Io)?;
    // Skip flags (3 bytes)
    let mut flags = [0u8; 3];
    reader.read_exact(&mut flags).map_err(DemuxError::Io)?;

    let (timescale, duration) = if version == 1 {
        // 64-bit version
        let _creation_time = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        let _modification_time = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        let timescale = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let duration = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        (timescale, duration)
    } else {
        // 32-bit version (version == 0)
        let _creation_time = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let _modification_time = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let timescale = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let duration = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as u64;
        (timescale, duration)
    };

    debug!("mvhd: timescale={}, duration={}", timescale, duration);

    Ok(MvhdBox {
        timescale,
        duration,
    })
}

// ─── mdhd Box ───────────────────────────────────────────────────────

/// Parsed mdhd (Media Header) box — per-track timescale and duration.
#[derive(Clone, Debug)]
pub struct MdhdBox {
    pub timescale: u32,
    pub duration: u64,
}

/// Parse an mdhd box. Reader must be at content start.
pub fn parse_mdhd<R: Read + Seek>(reader: &mut R) -> Result<MdhdBox, DemuxError> {
    let version = reader.read_u8().map_err(DemuxError::Io)?;
    let mut flags = [0u8; 3];
    reader.read_exact(&mut flags).map_err(DemuxError::Io)?;

    let (timescale, duration) = if version == 1 {
        let _creation_time = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        let _modification_time = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        let timescale = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let duration = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        (timescale, duration)
    } else {
        let _creation_time = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let _modification_time = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let timescale = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let duration = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as u64;
        (timescale, duration)
    };

    debug!("mdhd: timescale={}, duration={}", timescale, duration);

    Ok(MdhdBox {
        timescale,
        duration,
    })
}

// ─── hdlr Box ───────────────────────────────────────────────────────

/// Parsed hdlr (Handler Reference) box — identifies track type.
#[derive(Clone, Debug)]
pub struct HdlrBox {
    /// Handler type FourCC: 'vide', 'soun', etc.
    pub handler_type: u32,
    pub name: String,
}

/// Parse an hdlr box. Reader must be at content start.
pub fn parse_hdlr<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<HdlrBox, DemuxError> {
    let content_size = header
        .content_size()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "hdlr box has no definite size".into(),
        })?;

    // version (1) + flags (3) = 4 bytes
    let _version = reader.read_u8().map_err(DemuxError::Io)?;
    let mut flags = [0u8; 3];
    reader.read_exact(&mut flags).map_err(DemuxError::Io)?;

    // pre_defined (4 bytes, should be 0)
    let _pre_defined = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;

    // handler_type (4 bytes)
    let handler_type = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;

    // reserved (3 * 4 = 12 bytes)
    let mut reserved = [0u8; 12];
    reader.read_exact(&mut reserved).map_err(DemuxError::Io)?;

    // name (remaining bytes, null-terminated UTF-8)
    let name_len = content_size.saturating_sub(4 + 4 + 4 + 12) as usize;
    let mut name_buf = vec![0u8; name_len];
    reader.read_exact(&mut name_buf).map_err(DemuxError::Io)?;

    // Trim null terminator
    if let Some(pos) = name_buf.iter().position(|&b| b == 0) {
        name_buf.truncate(pos);
    }
    let name = String::from_utf8_lossy(&name_buf).to_string();

    debug!(
        "hdlr: handler_type='{}', name='{}'",
        fourcc_to_string(handler_type),
        name
    );

    Ok(HdlrBox { handler_type, name })
}

// ─── tkhd Box ───────────────────────────────────────────────────────

/// Parsed tkhd (Track Header) box — track id, dimensions.
#[derive(Clone, Debug)]
pub struct TkhdBox {
    pub track_id: u32,
    pub width: u32,
    pub height: u32,
    pub duration: u64,
}

/// Parse a tkhd box. Reader must be at content start.
pub fn parse_tkhd<R: Read + Seek>(reader: &mut R) -> Result<TkhdBox, DemuxError> {
    let version = reader.read_u8().map_err(DemuxError::Io)?;
    let mut flags = [0u8; 3];
    reader.read_exact(&mut flags).map_err(DemuxError::Io)?;

    let (track_id, duration) = if version == 1 {
        let _creation_time = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        let _modification_time = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        let track_id = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let _reserved = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let duration = reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?;
        (track_id, duration)
    } else {
        let _creation_time = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let _modification_time = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let track_id = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let _reserved = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let duration = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as u64;
        (track_id, duration)
    };

    // Skip: reserved (8), layer (2), alt_group (2), volume (2), reserved (2), matrix (36)
    let mut skip_buf = [0u8; 52];
    reader.read_exact(&mut skip_buf).map_err(DemuxError::Io)?;

    // width and height are 16.16 fixed-point
    let width_fp = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let height_fp = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let width = width_fp >> 16;
    let height = height_fp >> 16;

    debug!(
        "tkhd: track_id={}, duration={}, {}x{}",
        track_id, duration, width, height
    );

    Ok(TkhdBox {
        track_id,
        width,
        height,
        duration,
    })
}

// ─── stsd Box ───────────────────────────────────────────────────────

/// Video sample description extracted from stsd.
#[derive(Clone, Debug)]
pub struct VideoSampleDesc {
    /// Codec FourCC (avc1, hev1, etc.)
    pub codec_fourcc: u32,
    /// Width from sample description
    pub width: u16,
    /// Height from sample description
    pub height: u16,
    /// AVCC decoder configuration (SPS, PPS, etc.) — only for H.264
    pub avcc: Option<AvccConfig>,
    /// HVCC decoder configuration — only for H.265
    pub hvcc: Option<Vec<u8>>,
}

/// AVC Decoder Configuration Record (from avcC box inside stsd).
#[derive(Clone, Debug)]
pub struct AvccConfig {
    /// AVC profile indication
    pub profile: u8,
    /// Profile compatibility
    pub profile_compat: u8,
    /// AVC level indication
    pub level: u8,
    /// NAL unit length size minus one (typically 3, meaning 4-byte lengths)
    pub length_size_minus_one: u8,
    /// Sequence Parameter Sets
    pub sps_list: Vec<Vec<u8>>,
    /// Picture Parameter Sets
    pub pps_list: Vec<Vec<u8>>,
}

impl AvccConfig {
    /// The byte size used for NAL unit length fields in the bitstream.
    pub fn length_size(&self) -> u8 {
        self.length_size_minus_one + 1
    }
}

/// Result of parsing an stsd box — can be a video or audio sample description.
pub enum StsdResult {
    Video(VideoSampleDesc),
    Audio(AudioSampleDesc),
    None,
}

/// Parse stsd box. Reader must be at content start.
/// Returns the first video or audio sample description found.
pub fn parse_stsd<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<StsdResult, DemuxError> {
    let box_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "stsd box has no definite size".into(),
        })?;

    // version (1) + flags (3)
    let _version = reader.read_u8().map_err(DemuxError::Io)?;
    let mut flags = [0u8; 3];
    reader.read_exact(&mut flags).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;

    debug!("stsd: {} entries", entry_count);

    for _ in 0..entry_count {
        let entry_header =
            read_box_header(reader)?.ok_or_else(|| DemuxError::InvalidStructure {
                offset: reader.stream_position().unwrap_or(0),
                reason: "Unexpected EOF in stsd entries".into(),
            })?;

        match entry_header.box_type {
            AVC1 | AVC3 => {
                let desc = parse_avc_sample_entry(reader, &entry_header)?;
                return Ok(StsdResult::Video(desc));
            }
            HEV1 | HVC1 => {
                let desc = parse_hevc_sample_entry(reader, &entry_header)?;
                return Ok(StsdResult::Video(desc));
            }
            MP4A => {
                let desc = parse_mp4a_sample_entry(reader, &entry_header)?;
                return Ok(StsdResult::Audio(desc));
            }
            OPUS => {
                let desc = parse_opus_sample_entry(reader, &entry_header)?;
                return Ok(StsdResult::Audio(desc));
            }
            _ => {
                debug!(
                    "stsd: skipping unknown codec '{}'",
                    fourcc_to_string(entry_header.box_type)
                );
                skip_box(reader, &entry_header)?;
            }
        }
    }

    // Seek to end of stsd box
    reader
        .seek(SeekFrom::Start(box_end))
        .map_err(DemuxError::Io)?;

    Ok(StsdResult::None)
}

/// Parse an AVC (H.264) sample entry inside stsd.
fn parse_avc_sample_entry<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<VideoSampleDesc, DemuxError> {
    let entry_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "AVC sample entry has no definite size".into(),
        })?;

    // Skip: reserved (6), data_ref_index (2)
    let mut skip = [0u8; 8];
    reader.read_exact(&mut skip).map_err(DemuxError::Io)?;

    // Skip: pre_defined (2), reserved (2), pre_defined (12)
    let mut skip2 = [0u8; 16];
    reader.read_exact(&mut skip2).map_err(DemuxError::Io)?;

    let width = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;
    let height = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;

    // Skip: horiz_res (4), vert_res (4), reserved (4), frame_count (2),
    //        compressor_name (32), depth (2), pre_defined (2) = 50 bytes
    let mut skip3 = [0u8; 50];
    reader.read_exact(&mut skip3).map_err(DemuxError::Io)?;

    debug!("AVC sample entry: {}x{}", width, height);

    // Now parse sub-boxes until entry_end — looking for avcC
    let mut avcc = None;
    while reader.stream_position().map_err(DemuxError::Io)? < entry_end {
        let sub = match read_box_header(reader)? {
            Some(h) => h,
            None => break,
        };

        if sub.box_type == AVCC {
            avcc = Some(parse_avcc(reader, &sub)?);
        } else {
            skip_box(reader, &sub)?;
        }
    }

    // Seek to entry end to be safe
    reader
        .seek(SeekFrom::Start(entry_end))
        .map_err(DemuxError::Io)?;

    Ok(VideoSampleDesc {
        codec_fourcc: header.box_type,
        width,
        height,
        avcc,
        hvcc: None,
    })
}

/// Parse an HEVC (H.265) sample entry inside stsd.
fn parse_hevc_sample_entry<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<VideoSampleDesc, DemuxError> {
    let entry_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "HEVC sample entry has no definite size".into(),
        })?;

    // Skip: reserved (6), data_ref_index (2)
    let mut skip = [0u8; 8];
    reader.read_exact(&mut skip).map_err(DemuxError::Io)?;

    // Skip: pre_defined (2), reserved (2), pre_defined (12)
    let mut skip2 = [0u8; 16];
    reader.read_exact(&mut skip2).map_err(DemuxError::Io)?;

    let width = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;
    let height = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;

    // Skip: same as AVC (50 bytes)
    let mut skip3 = [0u8; 50];
    reader.read_exact(&mut skip3).map_err(DemuxError::Io)?;

    // Collect raw hvcC data
    let mut hvcc_data = None;
    while reader.stream_position().map_err(DemuxError::Io)? < entry_end {
        let sub = match read_box_header(reader)? {
            Some(h) => h,
            None => break,
        };

        if sub.box_type == HVCC {
            let content_size = sub.content_size().unwrap_or(0) as usize;
            let mut data = vec![0u8; content_size];
            reader.read_exact(&mut data).map_err(DemuxError::Io)?;
            hvcc_data = Some(data);
        } else {
            skip_box(reader, &sub)?;
        }
    }

    reader
        .seek(SeekFrom::Start(entry_end))
        .map_err(DemuxError::Io)?;

    Ok(VideoSampleDesc {
        codec_fourcc: header.box_type,
        width,
        height,
        avcc: None,
        hvcc: hvcc_data,
    })
}

// ─── avcC Box ───────────────────────────────────────────────────────

/// Parse an AVCDecoderConfigurationRecord from avcC box.
pub fn parse_avcc<R: Read + Seek>(
    reader: &mut R,
    _header: &BoxHeader,
) -> Result<AvccConfig, DemuxError> {
    let config_version = reader.read_u8().map_err(DemuxError::Io)?;
    if config_version != 1 {
        return Err(DemuxError::InvalidStructure {
            offset: reader.stream_position().unwrap_or(0),
            reason: format!("Unexpected avcC version: {}", config_version),
        });
    }

    let profile = reader.read_u8().map_err(DemuxError::Io)?;
    let profile_compat = reader.read_u8().map_err(DemuxError::Io)?;
    let level = reader.read_u8().map_err(DemuxError::Io)?;

    // length_size_minus_one is the lower 2 bits
    let ls_byte = reader.read_u8().map_err(DemuxError::Io)?;
    let length_size_minus_one = ls_byte & 0x03;

    // SPS count: lower 5 bits
    let sps_count_byte = reader.read_u8().map_err(DemuxError::Io)?;
    let sps_count = (sps_count_byte & 0x1F) as usize;

    let mut sps_list = Vec::with_capacity(sps_count);
    for _ in 0..sps_count {
        let sps_len = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)? as usize;
        let mut sps = vec![0u8; sps_len];
        reader.read_exact(&mut sps).map_err(DemuxError::Io)?;
        sps_list.push(sps);
    }

    // PPS count
    let pps_count = reader.read_u8().map_err(DemuxError::Io)? as usize;
    let mut pps_list = Vec::with_capacity(pps_count);
    for _ in 0..pps_count {
        let pps_len = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)? as usize;
        let mut pps = vec![0u8; pps_len];
        reader.read_exact(&mut pps).map_err(DemuxError::Io)?;
        pps_list.push(pps);
    }

    debug!(
        "avcC: profile={}, level={}, length_size={}, {} SPS, {} PPS",
        profile,
        level,
        length_size_minus_one + 1,
        sps_list.len(),
        pps_list.len()
    );

    Ok(AvccConfig {
        profile,
        profile_compat,
        level,
        length_size_minus_one,
        sps_list,
        pps_list,
    })
}

// ─── Audio Sample Entry Parsers ──────────────────────────────────────

/// Parse an mp4a (AAC) sample entry inside stsd.
///
/// ISO 14496-14 AudioSampleEntry layout:
/// - reserved (6 bytes)
/// - data_ref_index (2 bytes)
/// - reserved (8 bytes)  -- version(2), revision_level(2), vendor(4)
/// - channel_count (2 bytes)
/// - sample_size (2 bytes)
/// - compression_id (2 bytes)
/// - packet_size (2 bytes)
/// - sample_rate (4 bytes, 16.16 fixed-point)
///
/// Followed by sub-boxes, primarily esds.
fn parse_mp4a_sample_entry<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<AudioSampleDesc, DemuxError> {
    let entry_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "mp4a sample entry has no definite size".into(),
        })?;

    // reserved (6) + data_ref_index (2)
    let mut skip = [0u8; 8];
    reader.read_exact(&mut skip).map_err(DemuxError::Io)?;

    // Audio sample entry fields:
    // version (2) + revision_level (2) + vendor (4) = 8 bytes reserved
    let mut reserved = [0u8; 8];
    reader.read_exact(&mut reserved).map_err(DemuxError::Io)?;

    let channel_count = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;
    let sample_size = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;

    // compression_id (2) + packet_size (2)
    let mut skip2 = [0u8; 4];
    reader.read_exact(&mut skip2).map_err(DemuxError::Io)?;

    // sample_rate: 16.16 fixed-point
    let sample_rate_fp = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let sample_rate = sample_rate_fp >> 16;

    debug!(
        "mp4a sample entry: channels={}, sample_size={}, sample_rate={}",
        channel_count, sample_size, sample_rate
    );

    // Parse sub-boxes (esds, wave, etc.)
    let mut aac_config = None;
    while reader.stream_position().map_err(DemuxError::Io)? < entry_end {
        let sub = match read_box_header(reader)? {
            Some(h) => h,
            None => break,
        };

        match sub.box_type {
            ESDS => {
                aac_config = Some(parse_esds(reader, &sub)?);
            }
            WAVE => {
                // Some files wrap esds inside a wave box; search inside.
                let wave_end = sub.end_offset().unwrap_or(entry_end);
                while reader.stream_position().map_err(DemuxError::Io)? < wave_end {
                    let wave_child = match read_box_header(reader)? {
                        Some(h) => h,
                        None => break,
                    };
                    if wave_child.box_type == ESDS {
                        aac_config = Some(parse_esds(reader, &wave_child)?);
                    } else {
                        skip_box(reader, &wave_child)?;
                    }
                }
            }
            _ => {
                skip_box(reader, &sub)?;
            }
        }
    }

    // Seek to entry end
    reader
        .seek(SeekFrom::Start(entry_end))
        .map_err(DemuxError::Io)?;

    Ok(AudioSampleDesc {
        codec_fourcc: header.box_type,
        channel_count,
        sample_size,
        sample_rate,
        aac_config,
        opus_config: None,
    })
}

/// Parse an Opus sample entry inside stsd.
///
/// Opus in ISOBMFF: https://opus-codec.org/docs/opus_in_isobmff.html
/// OpusSampleEntry extends AudioSampleEntry, then contains dOps sub-box.
fn parse_opus_sample_entry<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<AudioSampleDesc, DemuxError> {
    let entry_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "Opus sample entry has no definite size".into(),
        })?;

    // reserved (6) + data_ref_index (2)
    let mut skip = [0u8; 8];
    reader.read_exact(&mut skip).map_err(DemuxError::Io)?;

    // Audio sample entry: version(2) + revision_level(2) + vendor(4)
    let mut reserved = [0u8; 8];
    reader.read_exact(&mut reserved).map_err(DemuxError::Io)?;

    let channel_count = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;
    let sample_size = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;

    // compression_id (2) + packet_size (2)
    let mut skip2 = [0u8; 4];
    reader.read_exact(&mut skip2).map_err(DemuxError::Io)?;

    // sample_rate (16.16 fixed-point) — for Opus this is always 48000
    let sample_rate_fp = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let sample_rate = sample_rate_fp >> 16;

    debug!(
        "Opus sample entry: channels={}, sample_size={}, sample_rate={}",
        channel_count, sample_size, sample_rate
    );

    // Parse sub-boxes — looking for dOps
    let mut opus_config = None;
    while reader.stream_position().map_err(DemuxError::Io)? < entry_end {
        let sub = match read_box_header(reader)? {
            Some(h) => h,
            None => break,
        };

        if sub.box_type == DOPS {
            opus_config = Some(parse_dops(reader, &sub)?);
        } else {
            skip_box(reader, &sub)?;
        }
    }

    // Seek to entry end
    reader
        .seek(SeekFrom::Start(entry_end))
        .map_err(DemuxError::Io)?;

    Ok(AudioSampleDesc {
        codec_fourcc: header.box_type,
        channel_count,
        sample_size,
        sample_rate,
        aac_config: None,
        opus_config,
    })
}

/// Parse an esds (Elementary Stream Descriptor) box to extract AAC config.
///
/// The esds box contains an ES_Descriptor (ISO 14496-1, section 8.3.3)
/// which wraps a DecoderConfigDescriptor containing DecoderSpecificInfo
/// (the AudioSpecificConfig for AAC).
fn parse_esds<R: Read + Seek>(reader: &mut R, header: &BoxHeader) -> Result<AacConfig, DemuxError> {
    let box_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "esds box has no definite size".into(),
        })?;

    // version (1) + flags (3)
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    // Read remaining esds content
    let remaining = (box_end - reader.stream_position().map_err(DemuxError::Io)?) as usize;
    let mut data = vec![0u8; remaining];
    reader.read_exact(&mut data).map_err(DemuxError::Io)?;

    // Parse ES_Descriptor hierarchy to find AudioSpecificConfig
    let config = parse_es_descriptor(&data)?;

    // Seek to box end
    reader
        .seek(SeekFrom::Start(box_end))
        .map_err(DemuxError::Io)?;

    Ok(config)
}

/// Parse the ES_Descriptor byte stream to extract AudioSpecificConfig.
///
/// Descriptor tag hierarchy:
/// ES_Descriptor (tag=3) → DecoderConfigDescriptor (tag=4) → DecoderSpecificInfo (tag=5)
fn parse_es_descriptor(data: &[u8]) -> Result<AacConfig, DemuxError> {
    let mut pos = 0;

    // ES_Descriptor tag (0x03)
    if pos >= data.len() || data[pos] != 0x03 {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: expected ES_Descriptor tag (0x03)".into(),
        });
    }
    pos += 1;
    let _es_len = read_descriptor_length(data, &mut pos);

    // ES_ID (2 bytes) + stream_priority (1 byte)
    if pos + 3 > data.len() {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: truncated ES_Descriptor".into(),
        });
    }
    pos += 3;

    // DecoderConfigDescriptor tag (0x04)
    if pos >= data.len() || data[pos] != 0x04 {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: expected DecoderConfigDescriptor tag (0x04)".into(),
        });
    }
    pos += 1;
    let _dec_len = read_descriptor_length(data, &mut pos);

    // objectTypeIndication (1) + streamType (1) + bufferSizeDB (3) +
    // maxBitrate (4) + avgBitrate (4) = 13 bytes
    if pos + 13 > data.len() {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: truncated DecoderConfigDescriptor".into(),
        });
    }
    pos += 13;

    // DecoderSpecificInfo tag (0x05)
    if pos >= data.len() || data[pos] != 0x05 {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: expected DecoderSpecificInfo tag (0x05)".into(),
        });
    }
    pos += 1;
    let spec_len = read_descriptor_length(data, &mut pos);

    if pos + spec_len > data.len() {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: truncated DecoderSpecificInfo".into(),
        });
    }

    let raw_config = data[pos..pos + spec_len].to_vec();

    // Parse AudioSpecificConfig (ISO 14496-3, Section 1.6.2.1)
    // First 5 bits: audioObjectType
    // Next bits: samplingFrequencyIndex (4 bits), channelConfiguration (4 bits)
    if raw_config.len() < 2 {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: "esds: AudioSpecificConfig too short".into(),
        });
    }

    let byte0 = raw_config[0];
    let byte1 = raw_config[1];

    let audio_object_type = (byte0 >> 3) & 0x1F;
    let sampling_frequency_index = ((byte0 & 0x07) << 1) | ((byte1 >> 7) & 0x01);
    let channel_config = (byte1 >> 3) & 0x0F;

    let sample_rate = if (sampling_frequency_index as usize) < AAC_SAMPLE_RATES.len() {
        AAC_SAMPLE_RATES[sampling_frequency_index as usize]
    } else {
        // Index 0x0F means explicit 24-bit sample rate follows in the config.
        // For now, fall back to 0 (caller should use the sample entry rate).
        0
    };

    debug!(
        "esds: AAC object_type={}, freq_idx={}, sample_rate={}, channels={}",
        audio_object_type, sampling_frequency_index, sample_rate, channel_config
    );

    Ok(AacConfig {
        audio_object_type,
        sampling_frequency_index,
        sample_rate,
        channel_config,
        raw_config,
    })
}

/// Read a variable-length descriptor size (ISO 14496-1, section 8.3.3).
/// Each byte contributes 7 bits; bit 7 indicates continuation.
fn read_descriptor_length(data: &[u8], pos: &mut usize) -> usize {
    let mut len: usize = 0;
    for _ in 0..4 {
        if *pos >= data.len() {
            break;
        }
        let b = data[*pos];
        *pos += 1;
        len = (len << 7) | (b & 0x7F) as usize;
        if b & 0x80 == 0 {
            break;
        }
    }
    len
}

/// Parse a dOps (Opus Decoder Configuration) box.
///
/// See https://opus-codec.org/docs/opus_in_isobmff.html#4.3
fn parse_dops<R: Read + Seek>(
    reader: &mut R,
    header: &BoxHeader,
) -> Result<OpusConfig, DemuxError> {
    let box_end = header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: header.offset,
            reason: "dOps box has no definite size".into(),
        })?;

    let version = reader.read_u8().map_err(DemuxError::Io)?;
    let output_channel_count = reader.read_u8().map_err(DemuxError::Io)?;
    let pre_skip = reader.read_u16::<BigEndian>().map_err(DemuxError::Io)?;
    let input_sample_rate = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let output_gain = reader.read_i16::<BigEndian>().map_err(DemuxError::Io)?;
    let channel_mapping_family = reader.read_u8().map_err(DemuxError::Io)?;

    // If channel_mapping_family > 0, there is additional channel mapping data
    // which we skip for now.

    debug!(
        "dOps: version={}, channels={}, pre_skip={}, input_rate={}, gain={}, mapping={}",
        version,
        output_channel_count,
        pre_skip,
        input_sample_rate,
        output_gain,
        channel_mapping_family
    );

    // Seek to box end (skip any remaining data like channel mapping table)
    reader
        .seek(SeekFrom::Start(box_end))
        .map_err(DemuxError::Io)?;

    Ok(OpusConfig {
        version,
        output_channel_count,
        pre_skip,
        input_sample_rate,
        output_gain,
        channel_mapping_family,
    })
}

// ─── Sample Table Boxes (stbl children) ─────────────────────────────

/// stts (Decoding Time to Sample) entry.
#[derive(Clone, Debug)]
pub struct SttsEntry {
    pub sample_count: u32,
    pub sample_delta: u32,
}

/// Parse stts box. Returns the entry list.
pub fn parse_stts<R: Read>(reader: &mut R) -> Result<Vec<SttsEntry>, DemuxError> {
    // version (1) + flags (3)
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as usize;
    let mut entries = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        let sample_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let sample_delta = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        entries.push(SttsEntry {
            sample_count,
            sample_delta,
        });
    }

    debug!("stts: {} entries", entries.len());
    Ok(entries)
}

/// ctts (Composition Time to Sample) entry.
#[derive(Clone, Debug)]
pub struct CttsEntry {
    pub sample_count: u32,
    /// Composition offset (can be negative in version 1).
    pub sample_offset: i32,
}

/// Parse ctts box. Returns the entry list.
pub fn parse_ctts<R: Read>(reader: &mut R) -> Result<Vec<CttsEntry>, DemuxError> {
    let version = reader.read_u8().map_err(DemuxError::Io)?;
    let mut flags = [0u8; 3];
    reader.read_exact(&mut flags).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as usize;
    let mut entries = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        let sample_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let sample_offset = if version == 0 {
            // Unsigned in version 0
            reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as i32
        } else {
            // Signed in version 1
            reader.read_i32::<BigEndian>().map_err(DemuxError::Io)?
        };
        entries.push(CttsEntry {
            sample_count,
            sample_offset,
        });
    }

    debug!("ctts: {} entries (version {})", entries.len(), version);
    Ok(entries)
}

/// stsc (Sample to Chunk) entry.
#[derive(Clone, Debug)]
pub struct StscEntry {
    /// First chunk number (1-based).
    pub first_chunk: u32,
    pub samples_per_chunk: u32,
    pub sample_description_index: u32,
}

/// Parse stsc box.
pub fn parse_stsc<R: Read>(reader: &mut R) -> Result<Vec<StscEntry>, DemuxError> {
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as usize;
    let mut entries = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        let first_chunk = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let samples_per_chunk = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        let sample_description_index = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
        entries.push(StscEntry {
            first_chunk,
            samples_per_chunk,
            sample_description_index,
        });
    }

    debug!("stsc: {} entries", entries.len());
    Ok(entries)
}

/// Parsed stsz (Sample Size) box.
#[derive(Clone, Debug)]
pub struct StszBox {
    /// If non-zero, all samples have this uniform size.
    pub default_sample_size: u32,
    /// Individual sample sizes (empty if default_sample_size > 0).
    pub sample_sizes: Vec<u32>,
    /// Total sample count.
    pub sample_count: u32,
}

/// Parse stsz box.
pub fn parse_stsz<R: Read>(reader: &mut R) -> Result<StszBox, DemuxError> {
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    let default_sample_size = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;
    let sample_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?;

    let sample_sizes = if default_sample_size == 0 {
        let mut sizes = Vec::with_capacity(sample_count as usize);
        for _ in 0..sample_count {
            sizes.push(reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?);
        }
        sizes
    } else {
        Vec::new()
    };

    debug!(
        "stsz: {} samples, default_size={}",
        sample_count, default_sample_size
    );

    Ok(StszBox {
        default_sample_size,
        sample_sizes,
        sample_count,
    })
}

/// Parse stco (Chunk Offset, 32-bit) box.
pub fn parse_stco<R: Read>(reader: &mut R) -> Result<Vec<u64>, DemuxError> {
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as usize;
    let mut offsets = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        offsets.push(reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as u64);
    }

    debug!("stco: {} chunk offsets", offsets.len());
    Ok(offsets)
}

/// Parse co64 (Chunk Offset, 64-bit) box.
pub fn parse_co64<R: Read>(reader: &mut R) -> Result<Vec<u64>, DemuxError> {
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as usize;
    let mut offsets = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        offsets.push(reader.read_u64::<BigEndian>().map_err(DemuxError::Io)?);
    }

    debug!("co64: {} chunk offsets", offsets.len());
    Ok(offsets)
}

/// Parse stss (Sync Sample / Keyframe) box. Returns 1-based sample numbers.
pub fn parse_stss<R: Read>(reader: &mut R) -> Result<Vec<u32>, DemuxError> {
    let mut vf = [0u8; 4];
    reader.read_exact(&mut vf).map_err(DemuxError::Io)?;

    let entry_count = reader.read_u32::<BigEndian>().map_err(DemuxError::Io)? as usize;
    let mut sync_samples = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        sync_samples.push(reader.read_u32::<BigEndian>().map_err(DemuxError::Io)?);
    }

    debug!("stss: {} sync samples", sync_samples.len());
    Ok(sync_samples)
}

// ─── Parsed Track ───────────────────────────────────────────────────

// ─── Audio Sample Description ────────────────────────────────────────

/// Audio sample description extracted from stsd (mp4a, Opus, etc.).
#[derive(Clone, Debug)]
pub struct AudioSampleDesc {
    /// Codec FourCC (mp4a, Opus, etc.)
    pub codec_fourcc: u32,
    /// Number of audio channels.
    pub channel_count: u16,
    /// Bits per sample (typically 16).
    pub sample_size: u16,
    /// Sample rate in Hz (from sample entry fixed-point field).
    pub sample_rate: u32,
    /// AAC AudioSpecificConfig (from esds box), if present.
    pub aac_config: Option<AacConfig>,
    /// Opus decoder config (from dOps box), if present.
    pub opus_config: Option<OpusConfig>,
}

/// AAC AudioSpecificConfig extracted from the esds box.
#[derive(Clone, Debug)]
pub struct AacConfig {
    /// AAC profile (1=AAC-LC, 2=HE-AAC, etc.).
    pub audio_object_type: u8,
    /// Sampling frequency index.
    pub sampling_frequency_index: u8,
    /// Actual sample rate in Hz (from the frequency table or explicit value).
    pub sample_rate: u32,
    /// Number of channels from config.
    pub channel_config: u8,
    /// Raw AudioSpecificConfig bytes for decoder initialization.
    pub raw_config: Vec<u8>,
}

/// Opus decoder configuration from the dOps box.
#[derive(Clone, Debug)]
pub struct OpusConfig {
    /// Opus version (should be 0).
    pub version: u8,
    /// Number of output channels.
    pub output_channel_count: u8,
    /// Pre-skip in samples at 48kHz.
    pub pre_skip: u16,
    /// Input sample rate (informational).
    pub input_sample_rate: u32,
    /// Output gain in dB (Q7.8 fixed-point).
    pub output_gain: i16,
    /// Channel mapping family (0 = mono/stereo, 1 = Vorbis mapping, 2+ = reserved).
    pub channel_mapping_family: u8,
}

/// AAC sampling frequency table (ISO 14496-3, Table 1.16).
const AAC_SAMPLE_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// All parsed data for a single video track.
#[derive(Clone, Debug)]
pub struct ParsedVideoTrack {
    pub track_id: u32,
    pub timescale: u32,
    pub duration: u64,
    pub width: u32,
    pub height: u32,
    pub sample_desc: VideoSampleDesc,
    pub stts: Vec<SttsEntry>,
    pub ctts: Vec<CttsEntry>,
    pub stsc: Vec<StscEntry>,
    pub stsz: StszBox,
    pub chunk_offsets: Vec<u64>,
    /// 1-based sync sample (keyframe) numbers. Empty means all samples are sync.
    pub sync_samples: Vec<u32>,
}

/// All parsed data for a single audio track.
#[derive(Clone, Debug)]
pub struct ParsedAudioTrack {
    pub track_id: u32,
    pub timescale: u32,
    pub duration: u64,
    pub sample_desc: AudioSampleDesc,
    pub stts: Vec<SttsEntry>,
    pub ctts: Vec<CttsEntry>,
    pub stsc: Vec<StscEntry>,
    pub stsz: StszBox,
    pub chunk_offsets: Vec<u64>,
    /// 1-based sync sample numbers. Empty means all samples are sync
    /// (typical for audio — all audio samples are random-access points).
    pub sync_samples: Vec<u32>,
}

/// All parsed data from the moov box.
#[derive(Clone, Debug)]
pub struct ParsedMoov {
    pub timescale: u32,
    pub duration: u64,
    pub video_tracks: Vec<ParsedVideoTrack>,
    pub audio_tracks: Vec<ParsedAudioTrack>,
}

/// Parse the entire moov box hierarchy: moov → trak → mdia → minf → stbl.
pub fn parse_moov<R: Read + Seek>(
    reader: &mut R,
    moov_header: &BoxHeader,
) -> Result<ParsedMoov, DemuxError> {
    let moov_end = moov_header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: moov_header.offset,
            reason: "moov box has no definite size".into(),
        })?;

    let mut mvhd_data: Option<MvhdBox> = None;
    let mut video_tracks = Vec::new();
    let mut audio_tracks = Vec::new();

    while reader.stream_position().map_err(DemuxError::Io)? < moov_end {
        let child = match read_box_header(reader)? {
            Some(h) => h,
            None => break,
        };

        match child.box_type {
            MVHD => {
                mvhd_data = Some(parse_mvhd(reader)?);
                // Seek to end of mvhd box
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            TRAK => match parse_trak(reader, &child)? {
                TrackParseResult::Video(vt) => video_tracks.push(vt),
                TrackParseResult::Audio(at) => audio_tracks.push(at),
                TrackParseResult::Other => {}
            },
            _ => {
                skip_box(reader, &child)?;
            }
        }
    }

    let mvhd = mvhd_data.ok_or_else(|| DemuxError::InvalidStructure {
        offset: moov_header.offset,
        reason: "No mvhd box found in moov".into(),
    })?;

    debug!(
        "moov: global timescale={}, duration={}, {} video tracks, {} audio tracks",
        mvhd.timescale,
        mvhd.duration,
        video_tracks.len(),
        audio_tracks.len()
    );

    Ok(ParsedMoov {
        timescale: mvhd.timescale,
        duration: mvhd.duration,
        video_tracks,
        audio_tracks,
    })
}

/// Result of parsing a trak box: can be video, audio, or other (subtitle, metadata, etc.).
enum TrackParseResult {
    Video(ParsedVideoTrack),
    Audio(ParsedAudioTrack),
    Other,
}

/// Parse a trak box. Returns a TrackParseResult indicating the track type.
fn parse_trak<R: Read + Seek>(
    reader: &mut R,
    trak_header: &BoxHeader,
) -> Result<TrackParseResult, DemuxError> {
    let trak_end = trak_header
        .end_offset()
        .ok_or_else(|| DemuxError::InvalidStructure {
            offset: trak_header.offset,
            reason: "trak box has no definite size".into(),
        })?;

    let mut tkhd_data: Option<TkhdBox> = None;
    let mut mdhd_data: Option<MdhdBox> = None;
    let mut handler_type: Option<u32> = None;
    let mut video_sample_desc: Option<VideoSampleDesc> = None;
    let mut audio_sample_desc: Option<AudioSampleDesc> = None;
    let mut stts_entries: Option<Vec<SttsEntry>> = None;
    let mut ctts_entries: Vec<CttsEntry> = Vec::new();
    let mut stsc_entries: Option<Vec<StscEntry>> = None;
    let mut stsz_data: Option<StszBox> = None;
    let mut chunk_offsets: Option<Vec<u64>> = None;
    let mut sync_samples: Vec<u32> = Vec::new();

    // Parse trak children recursively
    parse_trak_children(
        reader,
        trak_end,
        &mut tkhd_data,
        &mut mdhd_data,
        &mut handler_type,
        &mut video_sample_desc,
        &mut audio_sample_desc,
        &mut stts_entries,
        &mut ctts_entries,
        &mut stsc_entries,
        &mut stsz_data,
        &mut chunk_offsets,
        &mut sync_samples,
    )?;

    // Seek to trak end
    reader
        .seek(SeekFrom::Start(trak_end))
        .map_err(DemuxError::Io)?;

    match handler_type {
        Some(VIDE) => {
            let tkhd = tkhd_data.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing tkhd".into(),
            })?;

            let mdhd = mdhd_data.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing mdhd".into(),
            })?;

            let desc = video_sample_desc.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing sample description in stsd".into(),
            })?;

            let stts = stts_entries.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing stts".into(),
            })?;

            let stsc = stsc_entries.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing stsc".into(),
            })?;

            let stsz = stsz_data.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing stsz".into(),
            })?;

            let offsets = chunk_offsets.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Video track missing stco/co64".into(),
            })?;

            // Use tkhd dimensions, but fall back to stsd if tkhd has zeros
            let width = if tkhd.width > 0 {
                tkhd.width
            } else {
                desc.width as u32
            };
            let height = if tkhd.height > 0 {
                tkhd.height
            } else {
                desc.height as u32
            };

            Ok(TrackParseResult::Video(ParsedVideoTrack {
                track_id: tkhd.track_id,
                timescale: mdhd.timescale,
                duration: mdhd.duration,
                width,
                height,
                sample_desc: desc,
                stts,
                ctts: ctts_entries,
                stsc,
                stsz,
                chunk_offsets: offsets,
                sync_samples,
            }))
        }
        Some(SOUN) => {
            let tkhd = tkhd_data.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing tkhd".into(),
            })?;

            let mdhd = mdhd_data.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing mdhd".into(),
            })?;

            let desc = audio_sample_desc.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing sample description in stsd".into(),
            })?;

            let stts = stts_entries.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing stts".into(),
            })?;

            let stsc = stsc_entries.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing stsc".into(),
            })?;

            let stsz = stsz_data.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing stsz".into(),
            })?;

            let offsets = chunk_offsets.ok_or_else(|| DemuxError::InvalidStructure {
                offset: trak_header.offset,
                reason: "Audio track missing stco/co64".into(),
            })?;

            Ok(TrackParseResult::Audio(ParsedAudioTrack {
                track_id: tkhd.track_id,
                timescale: mdhd.timescale,
                duration: mdhd.duration,
                sample_desc: desc,
                stts,
                ctts: ctts_entries,
                stsc,
                stsz,
                chunk_offsets: offsets,
                sync_samples,
            }))
        }
        _ => Ok(TrackParseResult::Other),
    }
}

/// Recursively parse children of trak, mdia, minf, stbl containers.
#[allow(clippy::too_many_arguments)]
fn parse_trak_children<R: Read + Seek>(
    reader: &mut R,
    container_end: u64,
    tkhd_data: &mut Option<TkhdBox>,
    mdhd_data: &mut Option<MdhdBox>,
    handler_type: &mut Option<u32>,
    video_sample_desc: &mut Option<VideoSampleDesc>,
    audio_sample_desc: &mut Option<AudioSampleDesc>,
    stts_entries: &mut Option<Vec<SttsEntry>>,
    ctts_entries: &mut Vec<CttsEntry>,
    stsc_entries: &mut Option<Vec<StscEntry>>,
    stsz_data: &mut Option<StszBox>,
    chunk_offsets: &mut Option<Vec<u64>>,
    sync_samples: &mut Vec<u32>,
) -> Result<(), DemuxError> {
    while reader.stream_position().map_err(DemuxError::Io)? < container_end {
        let child = match read_box_header(reader)? {
            Some(h) => h,
            None => break,
        };

        // Don't read past the container boundary
        if child.offset >= container_end {
            break;
        }

        match child.box_type {
            TKHD => {
                *tkhd_data = Some(parse_tkhd(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            // Containers: recurse into them
            MDIA | MINF | STBL => {
                let child_end = child.end_offset().unwrap_or(container_end);
                parse_trak_children(
                    reader,
                    child_end,
                    tkhd_data,
                    mdhd_data,
                    handler_type,
                    video_sample_desc,
                    audio_sample_desc,
                    stts_entries,
                    ctts_entries,
                    stsc_entries,
                    stsz_data,
                    chunk_offsets,
                    sync_samples,
                )?;
            }
            MDHD => {
                *mdhd_data = Some(parse_mdhd(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            HDLR => {
                let hdlr = parse_hdlr(reader, &child)?;
                *handler_type = Some(hdlr.handler_type);
            }
            STSD => {
                let stsd_result = parse_stsd(reader, &child)?;
                match stsd_result {
                    StsdResult::Video(vsd) => *video_sample_desc = Some(vsd),
                    StsdResult::Audio(asd) => *audio_sample_desc = Some(asd),
                    StsdResult::None => {}
                }
            }
            STTS => {
                *stts_entries = Some(parse_stts(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            CTTS => {
                *ctts_entries = parse_ctts(reader)?;
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            STSC => {
                *stsc_entries = Some(parse_stsc(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            STSZ => {
                *stsz_data = Some(parse_stsz(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            STCO => {
                *chunk_offsets = Some(parse_stco(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            CO64 => {
                *chunk_offsets = Some(parse_co64(reader)?);
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            STSS => {
                *sync_samples = parse_stss(reader)?;
                if let Some(end) = child.end_offset() {
                    reader.seek(SeekFrom::Start(end)).map_err(DemuxError::Io)?;
                }
            }
            _ => {
                skip_box(reader, &child)?;
            }
        }
    }

    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper: build a box from fourcc + payload.
    fn make_box(fourcc: u32, payload: &[u8]) -> Vec<u8> {
        let size = (payload.len() + 8) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(&fourcc.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    /// Helper: build a box with 64-bit extended size.
    fn make_box_ext(fourcc: u32, payload: &[u8]) -> Vec<u8> {
        let size: u64 = (payload.len() + 16) as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_be_bytes()); // size32 = 1 means extended
        buf.extend_from_slice(&fourcc.to_be_bytes());
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn test_read_box_header_basic() {
        // 32-bit size box: size=20, type=ftyp, then 12 bytes payload
        let data = make_box(FTYP, &[0u8; 12]);
        let mut cursor = Cursor::new(&data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();

        assert_eq!(header.box_type, FTYP);
        assert_eq!(header.size, 20);
        assert_eq!(header.offset, 0);
        assert_eq!(header.header_size, 8);
        assert_eq!(header.content_offset(), 8);
        assert_eq!(header.content_size(), Some(12));
        assert_eq!(header.end_offset(), Some(20));
    }

    #[test]
    fn test_read_box_header_extended_size() {
        let data = make_box_ext(MOOV, &[0u8; 32]);
        let mut cursor = Cursor::new(&data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();

        assert_eq!(header.box_type, MOOV);
        assert_eq!(header.size, 48); // 32 payload + 16 header
        assert_eq!(header.header_size, 16);
        assert_eq!(header.content_offset(), 16);
        assert_eq!(header.content_size(), Some(32));
    }

    #[test]
    fn test_read_box_header_eof() {
        let data: &[u8] = &[];
        let mut cursor = Cursor::new(data);
        let result = read_box_header(&mut cursor).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fourcc_to_string() {
        assert_eq!(fourcc_to_string(FTYP), "ftyp");
        assert_eq!(fourcc_to_string(MOOV), "moov");
        assert_eq!(fourcc_to_string(STBL), "stbl");
    }

    #[test]
    fn test_parse_ftyp() {
        let isom = fourcc(b'i', b's', b'o', b'm');
        let mp41 = fourcc(b'm', b'p', b'4', b'1');

        let mut payload = Vec::new();
        payload.extend_from_slice(&isom.to_be_bytes()); // major_brand
        payload.extend_from_slice(&0u32.to_be_bytes()); // minor_version
        payload.extend_from_slice(&isom.to_be_bytes()); // compatible[0]
        payload.extend_from_slice(&mp41.to_be_bytes()); // compatible[1]

        let data = make_box(FTYP, &payload);
        let mut cursor = Cursor::new(&data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();
        let ftyp = parse_ftyp(&mut cursor, &header).unwrap();

        assert_eq!(ftyp.major_brand, isom);
        assert_eq!(ftyp.minor_version, 0);
        assert_eq!(ftyp.compatible_brands.len(), 2);
        assert_eq!(ftyp.compatible_brands[0], isom);
        assert_eq!(ftyp.compatible_brands[1], mp41);
    }

    #[test]
    fn test_parse_avcc() {
        // Build a minimal AVCDecoderConfigurationRecord
        let sps = vec![0x67, 0x42, 0xC0, 0x1E]; // SPS NAL (4 bytes)
        let pps = vec![0x68, 0xCE, 0x38, 0x80]; // PPS NAL (4 bytes)

        let mut payload = vec![
            1,    // configurationVersion
            0x42, // AVCProfileIndication (Baseline)
            0xC0, // profile_compatibility
            30,   // AVCLevelIndication (3.0)
            0xFF, // lengthSizeMinusOne = 3 (lower 2 bits), upper 6 bits are 1s
            0xE1, // numOfSPS = 1 (lower 5 bits), upper 3 bits are 1s
        ];
        payload.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        payload.extend_from_slice(&sps);
        payload.push(1); // numOfPPS = 1
        payload.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        payload.extend_from_slice(&pps);

        let data = make_box(AVCC, &payload);
        let mut cursor = Cursor::new(&data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();
        let avcc = parse_avcc(&mut cursor, &header).unwrap();

        assert_eq!(avcc.profile, 0x42);
        assert_eq!(avcc.profile_compat, 0xC0);
        assert_eq!(avcc.level, 30);
        assert_eq!(avcc.length_size_minus_one, 3);
        assert_eq!(avcc.length_size(), 4);
        assert_eq!(avcc.sps_list.len(), 1);
        assert_eq!(avcc.pps_list.len(), 1);
        assert_eq!(avcc.sps_list[0], sps);
        assert_eq!(avcc.pps_list[0], pps);
    }

    #[test]
    fn test_parse_stts() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&2u32.to_be_bytes()); // entry_count
                                                        // Entry 1: 100 samples, delta=512
        payload.extend_from_slice(&100u32.to_be_bytes());
        payload.extend_from_slice(&512u32.to_be_bytes());
        // Entry 2: 50 samples, delta=1024
        payload.extend_from_slice(&50u32.to_be_bytes());
        payload.extend_from_slice(&1024u32.to_be_bytes());

        let mut cursor = Cursor::new(&payload);
        let entries = parse_stts(&mut cursor).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sample_count, 100);
        assert_eq!(entries[0].sample_delta, 512);
        assert_eq!(entries[1].sample_count, 50);
        assert_eq!(entries[1].sample_delta, 1024);
    }

    #[test]
    fn test_parse_stsc() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&2u32.to_be_bytes()); // entry_count
                                                        // Entry 1: first_chunk=1, samples_per_chunk=10, desc_index=1
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&10u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        // Entry 2: first_chunk=5, samples_per_chunk=5, desc_index=1
        payload.extend_from_slice(&5u32.to_be_bytes());
        payload.extend_from_slice(&5u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());

        let mut cursor = Cursor::new(&payload);
        let entries = parse_stsc(&mut cursor).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].first_chunk, 1);
        assert_eq!(entries[0].samples_per_chunk, 10);
        assert_eq!(entries[1].first_chunk, 5);
        assert_eq!(entries[1].samples_per_chunk, 5);
    }

    #[test]
    fn test_parse_stsz() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&0u32.to_be_bytes()); // default_sample_size = 0 (variable)
        payload.extend_from_slice(&3u32.to_be_bytes()); // sample_count
        payload.extend_from_slice(&100u32.to_be_bytes()); // sample 1
        payload.extend_from_slice(&200u32.to_be_bytes()); // sample 2
        payload.extend_from_slice(&150u32.to_be_bytes()); // sample 3

        let mut cursor = Cursor::new(&payload);
        let stsz = parse_stsz(&mut cursor).unwrap();

        assert_eq!(stsz.default_sample_size, 0);
        assert_eq!(stsz.sample_count, 3);
        assert_eq!(stsz.sample_sizes, vec![100, 200, 150]);
    }

    #[test]
    fn test_parse_stsz_uniform() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&512u32.to_be_bytes()); // default_sample_size = 512
        payload.extend_from_slice(&100u32.to_be_bytes()); // sample_count

        let mut cursor = Cursor::new(&payload);
        let stsz = parse_stsz(&mut cursor).unwrap();

        assert_eq!(stsz.default_sample_size, 512);
        assert_eq!(stsz.sample_count, 100);
        assert!(stsz.sample_sizes.is_empty());
    }

    #[test]
    fn test_parse_stco() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&3u32.to_be_bytes()); // entry_count
        payload.extend_from_slice(&1000u32.to_be_bytes());
        payload.extend_from_slice(&2000u32.to_be_bytes());
        payload.extend_from_slice(&3000u32.to_be_bytes());

        let mut cursor = Cursor::new(&payload);
        let offsets = parse_stco(&mut cursor).unwrap();

        assert_eq!(offsets, vec![1000, 2000, 3000]);
    }

    #[test]
    fn test_parse_co64() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&2u32.to_be_bytes()); // entry_count
        payload.extend_from_slice(&0x100000000u64.to_be_bytes()); // > 4GB offset
        payload.extend_from_slice(&0x200000000u64.to_be_bytes());

        let mut cursor = Cursor::new(&payload);
        let offsets = parse_co64(&mut cursor).unwrap();

        assert_eq!(offsets, vec![0x100000000, 0x200000000]);
    }

    #[test]
    fn test_parse_stss() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        payload.extend_from_slice(&3u32.to_be_bytes()); // entry_count
        payload.extend_from_slice(&1u32.to_be_bytes()); // sample 1
        payload.extend_from_slice(&25u32.to_be_bytes()); // sample 25
        payload.extend_from_slice(&50u32.to_be_bytes()); // sample 50

        let mut cursor = Cursor::new(&payload);
        let sync = parse_stss(&mut cursor).unwrap();

        assert_eq!(sync, vec![1, 25, 50]);
    }

    #[test]
    fn test_parse_ctts() {
        let mut payload = Vec::new();
        payload.push(0); // version 0
        payload.extend_from_slice(&[0, 0, 0]); // flags
        payload.extend_from_slice(&2u32.to_be_bytes()); // entry_count
                                                        // Entry 1: count=5, offset=1024
        payload.extend_from_slice(&5u32.to_be_bytes());
        payload.extend_from_slice(&1024u32.to_be_bytes());
        // Entry 2: count=3, offset=2048
        payload.extend_from_slice(&3u32.to_be_bytes());
        payload.extend_from_slice(&2048u32.to_be_bytes());

        let mut cursor = Cursor::new(&payload);
        let entries = parse_ctts(&mut cursor).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sample_count, 5);
        assert_eq!(entries[0].sample_offset, 1024);
        assert_eq!(entries[1].sample_count, 3);
        assert_eq!(entries[1].sample_offset, 2048);
    }

    #[test]
    fn test_skip_box() {
        let data1 = make_box(FTYP, &[0u8; 20]);
        let data2 = make_box(MOOV, &[0u8; 30]);
        let mut combined = Vec::new();
        combined.extend_from_slice(&data1);
        combined.extend_from_slice(&data2);

        let mut cursor = Cursor::new(&combined);

        // Read first box header
        let h1 = read_box_header(&mut cursor).unwrap().unwrap();
        assert_eq!(h1.box_type, FTYP);

        // Skip it
        skip_box(&mut cursor, &h1).unwrap();

        // Should now be at the second box
        let h2 = read_box_header(&mut cursor).unwrap().unwrap();
        assert_eq!(h2.box_type, MOOV);
        assert_eq!(h2.offset, data1.len() as u64);
    }

    #[test]
    fn test_parse_mvhd_v0() {
        let mut payload = Vec::new();
        payload.push(0); // version
        payload.extend_from_slice(&[0, 0, 0]); // flags
        payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        payload.extend_from_slice(&90000u32.to_be_bytes()); // timescale
        payload.extend_from_slice(&(90000u32 * 10).to_be_bytes()); // duration (10 seconds)

        let mut cursor = Cursor::new(&payload);
        let mvhd = parse_mvhd(&mut cursor).unwrap();

        assert_eq!(mvhd.timescale, 90000);
        assert_eq!(mvhd.duration, 900000);
    }

    // ─── Audio parsing tests ───────────────────────────────────

    /// Build an esds box payload with a minimal AAC-LC AudioSpecificConfig.
    fn build_esds_payload(audio_object_type: u8, freq_idx: u8, channel_config: u8) -> Vec<u8> {
        // Build AudioSpecificConfig (2 bytes)
        let asc_byte0 = (audio_object_type << 3) | (freq_idx >> 1);
        let asc_byte1 = ((freq_idx & 0x01) << 7) | (channel_config << 3);
        let asc = [asc_byte0, asc_byte1];

        // Build ES_Descriptor
        let mut es_desc = Vec::new();
        // ES_ID (2) + stream_priority (1)
        es_desc.extend_from_slice(&[0x00, 0x01, 0x00]);

        // DecoderConfigDescriptor: tag=0x04
        let mut dec_config = Vec::new();
        // objectTypeIndication(1) + streamType(1) + bufferSizeDB(3) + maxBitrate(4) + avgBitrate(4)
        dec_config.push(0x40); // objectTypeIndication: Audio ISO/IEC 14496-3
        dec_config.push(0x15); // streamType: audio stream
        dec_config.extend_from_slice(&[0x00, 0x00, 0x00]); // bufferSizeDB
        dec_config.extend_from_slice(&128000u32.to_be_bytes()); // maxBitrate
        dec_config.extend_from_slice(&128000u32.to_be_bytes()); // avgBitrate

        // DecoderSpecificInfo: tag=0x05
        dec_config.push(0x05);
        dec_config.push(asc.len() as u8); // length
        dec_config.extend_from_slice(&asc);

        es_desc.push(0x04); // DecoderConfigDescriptor tag
        es_desc.push(dec_config.len() as u8);
        es_desc.extend_from_slice(&dec_config);

        // Wrap in ES_Descriptor tag=0x03
        let mut payload = Vec::new();
        // version(1) + flags(3)
        payload.extend_from_slice(&[0, 0, 0, 0]);
        payload.push(0x03); // ES_Descriptor tag
        payload.push(es_desc.len() as u8);
        payload.extend_from_slice(&es_desc);

        payload
    }

    #[test]
    fn test_parse_esds_aac_lc_44100() {
        // AAC-LC, 44100 Hz, stereo
        // audio_object_type=2 (AAC-LC), freq_idx=4 (44100), channels=2
        let payload = build_esds_payload(2, 4, 2);

        // We need to wrap in an esds box for parse_esds
        let box_data = make_box(ESDS, &payload);
        let mut cursor = Cursor::new(&box_data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();
        let config = parse_esds(&mut cursor, &header).unwrap();

        assert_eq!(config.audio_object_type, 2);
        assert_eq!(config.sampling_frequency_index, 4);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channel_config, 2);
        assert_eq!(config.raw_config.len(), 2);
    }

    #[test]
    fn test_parse_esds_aac_lc_48000() {
        // AAC-LC, 48000 Hz, stereo
        // audio_object_type=2 (AAC-LC), freq_idx=3 (48000), channels=2
        let payload = build_esds_payload(2, 3, 2);

        let box_data = make_box(ESDS, &payload);
        let mut cursor = Cursor::new(&box_data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();
        let config = parse_esds(&mut cursor, &header).unwrap();

        assert_eq!(config.audio_object_type, 2);
        assert_eq!(config.sampling_frequency_index, 3);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channel_config, 2);
    }

    #[test]
    fn test_parse_dops() {
        let mut payload = Vec::new();
        payload.push(0); // version
        payload.push(2); // output_channel_count (stereo)
        payload.extend_from_slice(&312u16.to_be_bytes()); // pre_skip
        payload.extend_from_slice(&48000u32.to_be_bytes()); // input_sample_rate
        payload.extend_from_slice(&0i16.to_be_bytes()); // output_gain
        payload.push(0); // channel_mapping_family

        let box_data = make_box(DOPS, &payload);
        let mut cursor = Cursor::new(&box_data);
        let header = read_box_header(&mut cursor).unwrap().unwrap();
        let config = parse_dops(&mut cursor, &header).unwrap();

        assert_eq!(config.version, 0);
        assert_eq!(config.output_channel_count, 2);
        assert_eq!(config.pre_skip, 312);
        assert_eq!(config.input_sample_rate, 48000);
        assert_eq!(config.output_gain, 0);
        assert_eq!(config.channel_mapping_family, 0);
    }

    #[test]
    fn test_parse_es_descriptor_basic() {
        // Build a minimal ES_Descriptor with AAC-LC 44100 stereo
        let asc = [0x12, 0x10]; // objectType=2, freqIdx=4, channels=2

        let mut data = Vec::new();
        // ES_Descriptor tag
        data.push(0x03);
        // length (will compute)
        let es_inner_start = data.len();
        data.push(0); // placeholder
                      // ES_ID + stream_priority
        data.extend_from_slice(&[0x00, 0x01, 0x00]);
        // DecoderConfigDescriptor
        data.push(0x04);
        let dec_inner_start = data.len();
        data.push(0); // placeholder
                      // 13 bytes of config
        data.push(0x40); // objectTypeIndication
        data.push(0x15); // streamType
        data.extend_from_slice(&[0x00, 0x00, 0x00]); // bufferSizeDB
        data.extend_from_slice(&[0x00, 0x01, 0xF4, 0x00]); // maxBitrate
        data.extend_from_slice(&[0x00, 0x01, 0xF4, 0x00]); // avgBitrate
                                                           // DecoderSpecificInfo
        data.push(0x05);
        data.push(asc.len() as u8);
        data.extend_from_slice(&asc);
        // Fix DecoderConfigDescriptor length
        let dec_len = data.len() - dec_inner_start - 1;
        data[dec_inner_start] = dec_len as u8;
        // Fix ES_Descriptor length
        let es_len = data.len() - es_inner_start - 1;
        data[es_inner_start] = es_len as u8;

        let config = parse_es_descriptor(&data).unwrap();
        assert_eq!(config.audio_object_type, 2);
        assert_eq!(config.sampling_frequency_index, 4);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channel_config, 2);
    }

    #[test]
    fn test_descriptor_length_single_byte() {
        let data = [0x25]; // length = 37
        let mut pos = 0;
        let len = read_descriptor_length(&data, &mut pos);
        assert_eq!(len, 37);
        assert_eq!(pos, 1);
    }

    #[test]
    fn test_descriptor_length_multi_byte() {
        // Two bytes: 0x80 | 0x01 = continuation, then 0x00 = 128
        let data = [0x81, 0x00]; // (1 << 7) | 0 = 128
        let mut pos = 0;
        let len = read_descriptor_length(&data, &mut pos);
        assert_eq!(len, 128);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_aac_sample_rates_table() {
        assert_eq!(AAC_SAMPLE_RATES[0], 96000);
        assert_eq!(AAC_SAMPLE_RATES[3], 48000);
        assert_eq!(AAC_SAMPLE_RATES[4], 44100);
        assert_eq!(AAC_SAMPLE_RATES[11], 8000);
    }
}
