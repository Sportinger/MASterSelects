//! NAL unit parser — AVCC to Annex-B conversion and SPS/PPS extraction.

/// NAL unit type for H.264.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum H264NalType {
    Slice,
    SliceA,
    SliceB,
    SliceC,
    Idr,
    Sei,
    Sps,
    Pps,
    Aud,
    EndSeq,
    EndStream,
    FillerData,
    Other(u8),
}

impl From<u8> for H264NalType {
    fn from(val: u8) -> Self {
        match val & 0x1F {
            1 => Self::Slice,
            2 => Self::SliceA,
            3 => Self::SliceB,
            4 => Self::SliceC,
            5 => Self::Idr,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::Aud,
            10 => Self::EndSeq,
            11 => Self::EndStream,
            12 => Self::FillerData,
            other => Self::Other(other),
        }
    }
}

/// Annex-B start code (4 bytes).
pub const ANNEXB_START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Convert AVCC-formatted NAL units to Annex-B format.
///
/// AVCC: `[length_size bytes length][NAL data]...`
/// Annex-B: `[0x00 0x00 0x00 0x01][NAL data]...`
pub fn avcc_to_annexb(avcc_data: &[u8], length_size: u8) -> Vec<u8> {
    let mut output = Vec::with_capacity(avcc_data.len() + 64);
    let ls = length_size as usize;
    let mut offset = 0;

    while offset + ls <= avcc_data.len() {
        // Read NAL unit length
        let nal_len = read_nal_length(&avcc_data[offset..], ls);
        offset += ls;

        if offset + nal_len > avcc_data.len() {
            break;
        }

        // Write Annex-B start code + NAL data
        output.extend_from_slice(&ANNEXB_START_CODE);
        output.extend_from_slice(&avcc_data[offset..offset + nal_len]);
        offset += nal_len;
    }

    output
}

/// Prepend SPS/PPS NAL units to a keyframe packet (Annex-B format).
pub fn prepend_sps_pps(sps: &[u8], pps: &[u8], frame_data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(sps.len() + pps.len() + frame_data.len() + 16);

    // SPS
    output.extend_from_slice(&ANNEXB_START_CODE);
    output.extend_from_slice(sps);

    // PPS
    output.extend_from_slice(&ANNEXB_START_CODE);
    output.extend_from_slice(pps);

    // Frame data (already contains start codes if Annex-B)
    output.extend_from_slice(frame_data);

    output
}

/// Read a variable-length NAL unit size (1, 2, 3, or 4 bytes big-endian).
fn read_nal_length(data: &[u8], length_size: usize) -> usize {
    let mut val: usize = 0;
    for &byte in &data[..length_size] {
        val = (val << 8) | byte as usize;
    }
    val
}

/// Extract the NAL unit type from the first byte of NAL data.
pub fn nal_unit_type(nal_first_byte: u8) -> H264NalType {
    H264NalType::from(nal_first_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcc_to_annexb_basic() {
        // AVCC with 4-byte length: length=5, then 5 bytes of NAL data
        let avcc = [0x00, 0x00, 0x00, 0x05, 0x67, 0x01, 0x02, 0x03, 0x04];
        let annexb = avcc_to_annexb(&avcc, 4);
        assert_eq!(&annexb[..4], &ANNEXB_START_CODE);
        assert_eq!(&annexb[4..], &[0x67, 0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn nal_type_parsing() {
        assert_eq!(nal_unit_type(0x67), H264NalType::Sps); // 0x67 & 0x1F = 7
        assert_eq!(nal_unit_type(0x68), H264NalType::Pps); // 0x68 & 0x1F = 8
        assert_eq!(nal_unit_type(0x65), H264NalType::Idr); // 0x65 & 0x1F = 5
        assert_eq!(nal_unit_type(0x41), H264NalType::Slice); // 0x41 & 0x1F = 1
    }

    #[test]
    fn prepend_sps_pps_works() {
        let sps = vec![0x67, 0x01, 0x02];
        let pps = vec![0x68, 0x03];
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xFF];

        let result = prepend_sps_pps(&sps, &pps, &frame);

        // SPS
        assert_eq!(&result[0..4], &ANNEXB_START_CODE);
        assert_eq!(&result[4..7], &sps[..]);
        // PPS
        assert_eq!(&result[7..11], &ANNEXB_START_CODE);
        assert_eq!(&result[11..13], &pps[..]);
        // Frame
        assert_eq!(&result[13..], &frame[..]);
    }
}
