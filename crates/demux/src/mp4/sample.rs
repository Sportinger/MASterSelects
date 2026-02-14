//! Sample table interpretation — maps sample indices to file byte offsets,
//! sizes, timestamps, and keyframe status.
//!
//! Uses the parsed stsc, stsz, stco/co64, stts, ctts, and stss data
//! to build a flat index of samples for efficient random access.

use crate::mp4::boxes::{CttsEntry, ParsedAudioTrack, ParsedVideoTrack, StscEntry, SttsEntry};
use ms_common::DemuxError;
use tracing::debug;

/// Flat entry for a single sample, pre-computed for fast access.
#[derive(Clone, Debug)]
pub struct SampleEntry {
    /// 0-based sample index.
    pub index: u32,
    /// Byte offset in the file where this sample's data starts.
    pub offset: u64,
    /// Byte size of the sample data.
    pub size: u32,
    /// Decoding timestamp in media timescale units.
    pub dts: u64,
    /// Composition (presentation) timestamp in media timescale units.
    pub cts: i64,
    /// Whether this sample is a sync sample (keyframe).
    pub is_sync: bool,
}

/// Pre-computed sample table for a video track.
#[derive(Clone, Debug)]
pub struct SampleTable {
    /// Flat list of all samples, ordered by sample index (decode order).
    pub samples: Vec<SampleEntry>,
    /// Media timescale (ticks per second).
    pub timescale: u32,
    /// Total duration in timescale units.
    pub duration: u64,
}

impl SampleTable {
    /// Build a `SampleTable` from parsed track data.
    pub fn build(track: &ParsedVideoTrack) -> Result<Self, DemuxError> {
        let sample_count = track.stsz.sample_count as usize;
        if sample_count == 0 {
            return Ok(SampleTable {
                samples: Vec::new(),
                timescale: track.timescale,
                duration: track.duration,
            });
        }

        // Step 1: Build sample-to-file-offset mapping from stsc + stco + stsz
        let offsets_and_sizes = build_sample_offsets(
            &track.stsc,
            &track.chunk_offsets,
            &track.stsz.sample_sizes,
            track.stsz.default_sample_size,
            sample_count,
        )?;

        // Step 2: Build DTS array from stts
        let dts_array = build_dts_array(&track.stts, sample_count)?;

        // Step 3: Build composition offsets from ctts (if present)
        let cts_offsets = if track.ctts.is_empty() {
            vec![0i32; sample_count]
        } else {
            build_cts_offsets(&track.ctts, sample_count)?
        };

        // Step 4: Build sync sample set from stss
        let sync_set = build_sync_set(&track.sync_samples, sample_count);

        // Step 5: Assemble flat sample entries
        let mut samples = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let (offset, size) = offsets_and_sizes[i];
            let dts = dts_array[i];
            let cts = dts as i64 + cts_offsets[i] as i64;
            let is_sync = sync_set.is_empty() || sync_set.contains(&(i as u32 + 1));

            samples.push(SampleEntry {
                index: i as u32,
                offset,
                size,
                dts,
                cts,
                is_sync,
            });
        }

        debug!(
            "SampleTable: {} samples, timescale={}, duration={}",
            samples.len(),
            track.timescale,
            track.duration
        );

        Ok(SampleTable {
            samples,
            timescale: track.timescale,
            duration: track.duration,
        })
    }

    /// Build a `SampleTable` from a parsed audio track.
    ///
    /// Audio sample tables use the same MP4 box infrastructure as video
    /// (stts, stsc, stsz, stco/co64). The main difference is that audio
    /// tracks typically have no stss box (all samples are sync) and the
    /// timescale is usually 44100 or 48000 Hz.
    pub fn build_from_audio(track: &ParsedAudioTrack) -> Result<Self, DemuxError> {
        let sample_count = track.stsz.sample_count as usize;
        if sample_count == 0 {
            return Ok(SampleTable {
                samples: Vec::new(),
                timescale: track.timescale,
                duration: track.duration,
            });
        }

        // Step 1: Build sample-to-file-offset mapping
        let offsets_and_sizes = build_sample_offsets(
            &track.stsc,
            &track.chunk_offsets,
            &track.stsz.sample_sizes,
            track.stsz.default_sample_size,
            sample_count,
        )?;

        // Step 2: Build DTS array from stts
        let dts_array = build_dts_array(&track.stts, sample_count)?;

        // Step 3: Build composition offsets (usually empty for audio)
        let cts_offsets = if track.ctts.is_empty() {
            vec![0i32; sample_count]
        } else {
            build_cts_offsets(&track.ctts, sample_count)?
        };

        // Step 4: Build sync sample set
        // For audio, stss is typically absent (all samples are sync)
        let sync_set = build_sync_set(&track.sync_samples, sample_count);

        // Step 5: Assemble flat sample entries
        let mut samples = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let (offset, size) = offsets_and_sizes[i];
            let dts = dts_array[i];
            let cts = dts as i64 + cts_offsets[i] as i64;
            let is_sync = sync_set.is_empty() || sync_set.contains(&(i as u32 + 1));

            samples.push(SampleEntry {
                index: i as u32,
                offset,
                size,
                dts,
                cts,
                is_sync,
            });
        }

        debug!(
            "SampleTable (audio): {} samples, timescale={}, duration={}",
            samples.len(),
            track.timescale,
            track.duration
        );

        Ok(SampleTable {
            samples,
            timescale: track.timescale,
            duration: track.duration,
        })
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        self.duration as f64 / self.timescale as f64
    }

    /// Convert a DTS/CTS in timescale units to seconds.
    pub fn ticks_to_secs(&self, ticks: i64) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        ticks as f64 / self.timescale as f64
    }

    /// Find the sync sample (keyframe) at or before the given time in seconds.
    /// Returns the sample index, or None if no sync sample is found.
    pub fn find_sync_at_or_before(&self, time_secs: f64) -> Option<usize> {
        if self.samples.is_empty() {
            return None;
        }

        let target_ticks = (time_secs * self.timescale as f64) as i64;

        // Find the last sample whose CTS <= target_ticks and is_sync
        let mut best: Option<usize> = None;
        for (i, s) in self.samples.iter().enumerate() {
            if s.is_sync && s.cts <= target_ticks {
                best = Some(i);
            }
            // Since DTS is monotonically increasing and CTS can vary,
            // we can't early-exit here. But in practice stss is sorted,
            // so we could optimize. For now, a full scan is fine.
        }

        best
    }

    /// Find the sample index for a given presentation time in seconds.
    /// Returns the sample whose CTS is closest to the given time.
    pub fn find_sample_at_time(&self, time_secs: f64) -> Option<usize> {
        if self.samples.is_empty() {
            return None;
        }

        let target_ticks = (time_secs * self.timescale as f64) as i64;
        let mut best_idx = 0;
        let mut best_diff = i64::MAX;

        for (i, s) in self.samples.iter().enumerate() {
            let diff = (s.cts - target_ticks).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        Some(best_idx)
    }
}

/// Build a vec of (file_offset, size) for each sample, using stsc + stco + stsz.
///
/// The stsc table maps chunk ranges to samples-per-chunk counts.
/// Combined with chunk offsets and sample sizes, we can compute
/// the exact byte offset for each sample.
fn build_sample_offsets(
    stsc: &[StscEntry],
    chunk_offsets: &[u64],
    sample_sizes: &[u32],
    default_sample_size: u32,
    sample_count: usize,
) -> Result<Vec<(u64, u32)>, DemuxError> {
    let mut result = Vec::with_capacity(sample_count);
    let mut sample_idx: usize = 0;

    for (chunk_idx, &chunk_offset) in chunk_offsets.iter().enumerate() {
        // chunk numbers are 1-based in stsc
        let chunk_num = chunk_idx as u32 + 1;

        // Find the applicable stsc entry for this chunk
        let samples_in_chunk = samples_per_chunk_for(stsc, chunk_num);

        let mut offset = chunk_offset;

        for _ in 0..samples_in_chunk {
            if sample_idx >= sample_count {
                break;
            }

            let size = if default_sample_size > 0 {
                default_sample_size
            } else if sample_idx < sample_sizes.len() {
                sample_sizes[sample_idx]
            } else {
                return Err(DemuxError::InvalidStructure {
                    offset: 0,
                    reason: format!(
                        "Sample index {} exceeds stsz table length {}",
                        sample_idx,
                        sample_sizes.len()
                    ),
                });
            };

            result.push((offset, size));
            offset += size as u64;
            sample_idx += 1;
        }
    }

    if result.len() != sample_count {
        return Err(DemuxError::InvalidStructure {
            offset: 0,
            reason: format!(
                "Built {} sample offsets but expected {} (stsc/stco/stsz mismatch)",
                result.len(),
                sample_count
            ),
        });
    }

    Ok(result)
}

/// Determine how many samples are in the given chunk (1-based chunk number),
/// based on the stsc (sample-to-chunk) entries.
fn samples_per_chunk_for(stsc: &[StscEntry], chunk_num: u32) -> u32 {
    // stsc entries define ranges: each entry applies from first_chunk
    // until the next entry's first_chunk - 1.
    // We find the last entry whose first_chunk <= chunk_num.
    let mut spc = 1; // default fallback
    for entry in stsc {
        if entry.first_chunk <= chunk_num {
            spc = entry.samples_per_chunk;
        } else {
            break;
        }
    }
    spc
}

/// Build the DTS (decoding timestamp) array from stts entries.
/// Returns a vec of DTS values in timescale units, one per sample.
fn build_dts_array(stts: &[SttsEntry], sample_count: usize) -> Result<Vec<u64>, DemuxError> {
    let mut dts_array = Vec::with_capacity(sample_count);
    let mut dts: u64 = 0;

    for entry in stts {
        for _ in 0..entry.sample_count {
            if dts_array.len() >= sample_count {
                break;
            }
            dts_array.push(dts);
            dts += entry.sample_delta as u64;
        }
    }

    if dts_array.len() < sample_count {
        // Pad with last delta if stts doesn't cover all samples
        let last_delta = stts.last().map(|e| e.sample_delta as u64).unwrap_or(1);
        while dts_array.len() < sample_count {
            dts_array.push(dts);
            dts += last_delta;
        }
    }

    Ok(dts_array)
}

/// Build composition time offset array from ctts entries.
/// Returns one offset per sample.
fn build_cts_offsets(ctts: &[CttsEntry], sample_count: usize) -> Result<Vec<i32>, DemuxError> {
    let mut offsets = Vec::with_capacity(sample_count);

    for entry in ctts {
        for _ in 0..entry.sample_count {
            if offsets.len() >= sample_count {
                break;
            }
            offsets.push(entry.sample_offset);
        }
    }

    // Pad with 0 if ctts doesn't cover all samples
    while offsets.len() < sample_count {
        offsets.push(0);
    }

    Ok(offsets)
}

/// Build a set of sync sample numbers (1-based) for fast lookup.
/// Returns empty set if stss is empty (meaning all samples are sync).
fn build_sync_set(sync_samples: &[u32], _sample_count: usize) -> Vec<u32> {
    // We keep this as a sorted Vec for simplicity.
    // For large files, a HashSet would be faster, but Vec with binary_search
    // is cache-friendly and typically stss is not huge.
    sync_samples.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4::boxes::{AvccConfig, StszBox, VideoSampleDesc, AVC1};

    /// Helper to create a minimal ParsedVideoTrack for testing.
    fn make_test_track(
        stts: Vec<SttsEntry>,
        ctts: Vec<CttsEntry>,
        stsc: Vec<StscEntry>,
        sample_sizes: Vec<u32>,
        chunk_offsets: Vec<u64>,
        sync_samples: Vec<u32>,
    ) -> ParsedVideoTrack {
        let sample_count = sample_sizes.len() as u32;
        ParsedVideoTrack {
            track_id: 1,
            timescale: 30000,
            duration: 0, // computed from stts
            width: 1920,
            height: 1080,
            sample_desc: VideoSampleDesc {
                codec_fourcc: AVC1,
                width: 1920,
                height: 1080,
                avcc: Some(AvccConfig {
                    profile: 0x42,
                    profile_compat: 0xC0,
                    level: 30,
                    length_size_minus_one: 3,
                    sps_list: vec![vec![0x67, 0x42, 0xC0, 0x1E]],
                    pps_list: vec![vec![0x68, 0xCE, 0x38, 0x80]],
                }),
                hvcc: None,
            },
            stts,
            ctts,
            stsc,
            stsz: StszBox {
                default_sample_size: 0,
                sample_sizes,
                sample_count,
            },
            chunk_offsets,
            sync_samples,
        }
    }

    #[test]
    fn test_sample_table_basic() {
        // 1 chunk with 3 samples, each 100 bytes, chunk starts at offset 1000
        let track = make_test_track(
            vec![SttsEntry {
                sample_count: 3,
                sample_delta: 1001,
            }],
            vec![],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 3,
                sample_description_index: 1,
            }],
            vec![100, 200, 150],
            vec![1000],
            vec![1], // only sample 1 is a keyframe
        );

        let table = SampleTable::build(&track).unwrap();
        assert_eq!(table.samples.len(), 3);

        // Sample 0: offset=1000, size=100
        assert_eq!(table.samples[0].offset, 1000);
        assert_eq!(table.samples[0].size, 100);
        assert_eq!(table.samples[0].dts, 0);
        assert!(table.samples[0].is_sync);

        // Sample 1: offset=1100, size=200
        assert_eq!(table.samples[1].offset, 1100);
        assert_eq!(table.samples[1].size, 200);
        assert_eq!(table.samples[1].dts, 1001);
        assert!(!table.samples[1].is_sync);

        // Sample 2: offset=1300, size=150
        assert_eq!(table.samples[2].offset, 1300);
        assert_eq!(table.samples[2].size, 150);
        assert_eq!(table.samples[2].dts, 2002);
        assert!(!table.samples[2].is_sync);
    }

    #[test]
    fn test_sample_table_multiple_chunks() {
        // 2 chunks: chunk 1 has 2 samples, chunk 2 has 1 sample
        let track = make_test_track(
            vec![SttsEntry {
                sample_count: 3,
                sample_delta: 512,
            }],
            vec![],
            vec![
                StscEntry {
                    first_chunk: 1,
                    samples_per_chunk: 2,
                    sample_description_index: 1,
                },
                StscEntry {
                    first_chunk: 2,
                    samples_per_chunk: 1,
                    sample_description_index: 1,
                },
            ],
            vec![100, 200, 300],
            vec![1000, 5000],
            vec![], // all sync
        );

        let table = SampleTable::build(&track).unwrap();
        assert_eq!(table.samples.len(), 3);

        // Chunk 1 samples
        assert_eq!(table.samples[0].offset, 1000);
        assert_eq!(table.samples[0].size, 100);
        assert_eq!(table.samples[1].offset, 1100);
        assert_eq!(table.samples[1].size, 200);

        // Chunk 2 sample
        assert_eq!(table.samples[2].offset, 5000);
        assert_eq!(table.samples[2].size, 300);

        // All are sync since stss is empty
        assert!(table.samples[0].is_sync);
        assert!(table.samples[1].is_sync);
        assert!(table.samples[2].is_sync);
    }

    #[test]
    fn test_sample_table_with_ctts() {
        let track = make_test_track(
            vec![SttsEntry {
                sample_count: 4,
                sample_delta: 1000,
            }],
            vec![
                CttsEntry {
                    sample_count: 2,
                    sample_offset: 2000,
                },
                CttsEntry {
                    sample_count: 2,
                    sample_offset: 1000,
                },
            ],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 4,
                sample_description_index: 1,
            }],
            vec![100, 100, 100, 100],
            vec![1000],
            vec![1],
        );

        let table = SampleTable::build(&track).unwrap();
        assert_eq!(table.samples.len(), 4);

        // CTS = DTS + composition_offset
        assert_eq!(table.samples[0].dts, 0);
        assert_eq!(table.samples[0].cts, 2000); // 0 + 2000
        assert_eq!(table.samples[1].dts, 1000);
        assert_eq!(table.samples[1].cts, 3000); // 1000 + 2000
        assert_eq!(table.samples[2].dts, 2000);
        assert_eq!(table.samples[2].cts, 3000); // 2000 + 1000
        assert_eq!(table.samples[3].dts, 3000);
        assert_eq!(table.samples[3].cts, 4000); // 3000 + 1000
    }

    #[test]
    fn test_find_sync_at_or_before() {
        let track = make_test_track(
            vec![SttsEntry {
                sample_count: 10,
                sample_delta: 30000, // 1 second per sample at 30000 timescale
            }],
            vec![],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 10,
                sample_description_index: 1,
            }],
            vec![100; 10],
            vec![0],
            vec![1, 4, 7], // samples 1, 4, 7 are keyframes (1-based)
        );

        let table = SampleTable::build(&track).unwrap();

        // At 0.0s, should find sample 0 (sample 1 in 1-based)
        assert_eq!(table.find_sync_at_or_before(0.0), Some(0));

        // At 2.5s, should find sample 3 (sample 4 in 1-based, which is at DTS=3s)
        // Actually sample 3 is at DTS=3*30000/30000=3.0s, which is > 2.5
        // sample 0 is at 0.0s — that's the last sync before 2.5
        assert_eq!(table.find_sync_at_or_before(2.5), Some(0));

        // At 5.0s, should find sample 3 (at 3.0s, 1-based sample 4)
        assert_eq!(table.find_sync_at_or_before(5.0), Some(3));

        // At 8.0s, should find sample 6 (at 6.0s, 1-based sample 7)
        assert_eq!(table.find_sync_at_or_before(8.0), Some(6));
    }

    #[test]
    fn test_empty_sample_table() {
        let track = make_test_track(vec![], vec![], vec![], vec![], vec![], vec![]);

        let table = SampleTable::build(&track).unwrap();
        assert!(table.samples.is_empty());
        assert_eq!(table.find_sync_at_or_before(0.0), None);
    }

    #[test]
    fn test_ticks_to_secs() {
        let table = SampleTable {
            samples: vec![],
            timescale: 48000,
            duration: 96000,
        };

        assert!((table.ticks_to_secs(48000) - 1.0).abs() < 1e-9);
        assert!((table.duration_secs() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_uniform_sample_size() {
        // Test with default_sample_size (all same size)
        let mut track = make_test_track(
            vec![SttsEntry {
                sample_count: 3,
                sample_delta: 1000,
            }],
            vec![],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 3,
                sample_description_index: 1,
            }],
            vec![], // empty because uniform
            vec![2000],
            vec![],
        );
        track.stsz = StszBox {
            default_sample_size: 256,
            sample_sizes: vec![],
            sample_count: 3,
        };

        let table = SampleTable::build(&track).unwrap();
        assert_eq!(table.samples.len(), 3);
        assert_eq!(table.samples[0].offset, 2000);
        assert_eq!(table.samples[0].size, 256);
        assert_eq!(table.samples[1].offset, 2256);
        assert_eq!(table.samples[1].size, 256);
        assert_eq!(table.samples[2].offset, 2512);
        assert_eq!(table.samples[2].size, 256);
    }

    // ─── Audio sample table tests ───────────────────────────────

    /// Helper to create a minimal ParsedAudioTrack for testing.
    fn make_test_audio_track(
        stts: Vec<SttsEntry>,
        stsc: Vec<StscEntry>,
        sample_sizes: Vec<u32>,
        chunk_offsets: Vec<u64>,
    ) -> ParsedAudioTrack {
        use crate::mp4::boxes::{AudioSampleDesc, MP4A};
        let sample_count = sample_sizes.len() as u32;
        ParsedAudioTrack {
            track_id: 2,
            timescale: 44100,
            duration: 0,
            sample_desc: AudioSampleDesc {
                codec_fourcc: MP4A,
                channel_count: 2,
                sample_size: 16,
                sample_rate: 44100,
                aac_config: None,
                opus_config: None,
            },
            stts,
            ctts: vec![],
            stsc,
            stsz: StszBox {
                default_sample_size: 0,
                sample_sizes,
                sample_count,
            },
            chunk_offsets,
            sync_samples: vec![], // all audio samples are sync
        }
    }

    #[test]
    fn test_audio_sample_table_basic() {
        // 1 chunk with 3 AAC samples (1024 frames each at 44100Hz)
        let track = make_test_audio_track(
            vec![SttsEntry {
                sample_count: 3,
                sample_delta: 1024,
            }],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 3,
                sample_description_index: 1,
            }],
            vec![400, 380, 410],
            vec![5000],
        );

        let table = SampleTable::build_from_audio(&track).unwrap();
        assert_eq!(table.samples.len(), 3);
        assert_eq!(table.timescale, 44100);

        // Sample 0: offset=5000, size=400
        assert_eq!(table.samples[0].offset, 5000);
        assert_eq!(table.samples[0].size, 400);
        assert_eq!(table.samples[0].dts, 0);
        assert!(table.samples[0].is_sync); // all audio samples are sync

        // Sample 1: offset=5400, size=380
        assert_eq!(table.samples[1].offset, 5400);
        assert_eq!(table.samples[1].size, 380);
        assert_eq!(table.samples[1].dts, 1024);
        assert!(table.samples[1].is_sync);

        // Sample 2: offset=5780, size=410
        assert_eq!(table.samples[2].offset, 5780);
        assert_eq!(table.samples[2].size, 410);
        assert_eq!(table.samples[2].dts, 2048);
        assert!(table.samples[2].is_sync);
    }

    #[test]
    fn test_audio_sample_table_timestamps() {
        // AAC at 44100Hz: each sample is 1024 frames
        let track = make_test_audio_track(
            vec![SttsEntry {
                sample_count: 5,
                sample_delta: 1024,
            }],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 5,
                sample_description_index: 1,
            }],
            vec![400; 5],
            vec![1000],
        );

        let table = SampleTable::build_from_audio(&track).unwrap();

        // Verify timestamps: 1024 samples at 44100Hz = ~23.2ms per sample
        let pts0 = table.ticks_to_secs(table.samples[0].cts);
        let pts1 = table.ticks_to_secs(table.samples[1].cts);
        let expected_delta = 1024.0 / 44100.0;

        assert!((pts0 - 0.0).abs() < 1e-9);
        assert!((pts1 - expected_delta).abs() < 1e-6);
    }

    #[test]
    fn test_audio_sample_table_empty() {
        let mut track = make_test_audio_track(vec![], vec![], vec![], vec![]);
        track.stsz.sample_count = 0;

        let table = SampleTable::build_from_audio(&track).unwrap();
        assert!(table.samples.is_empty());
    }

    #[test]
    fn test_audio_find_sample_at_time() {
        // 10 audio samples, each 1024 frames at 44100Hz
        let track = make_test_audio_track(
            vec![SttsEntry {
                sample_count: 10,
                sample_delta: 1024,
            }],
            vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: 10,
                sample_description_index: 1,
            }],
            vec![400; 10],
            vec![1000],
        );

        let table = SampleTable::build_from_audio(&track).unwrap();

        // At 0.0s, should be sample 0
        assert_eq!(table.find_sample_at_time(0.0), Some(0));

        // At 0.023s (just past first sample), should be sample 1
        assert_eq!(table.find_sample_at_time(0.024), Some(1));
    }
}
