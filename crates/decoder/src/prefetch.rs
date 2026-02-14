//! Prefetch Queue — look-ahead frame decoding for smooth playback.
//!
//! During playback the render thread consumes one frame per vsync.
//! The `PrefetchQueue` sits between the decoder and the compositor:
//! it pre-decodes several frames ahead of the playhead so that a
//! decoded frame is always ready when the compositor asks for it.
//!
//! The queue maintains a ring buffer of `PrefetchFrame` entries.
//! Each entry starts in `Pending` state, transitions to `Rgba`
//! (or a GPU texture handle in later phases) once the decode thread
//! has finished, and is eventually consumed and discarded as the
//! playhead advances.
//!
//! On a seek the queue is cleared and new requests are issued.

use std::collections::VecDeque;

use ms_common::{FrameNumber, SourceId, TimeCode};

/// The decoded (or pending) payload of a prefetched frame.
#[derive(Clone, Debug)]
pub enum PrefetchData {
    /// RGBA CPU-side pixel data (Phase 0 / software fallback).
    Rgba(Vec<u8>, u32, u32),
    /// Decode not yet complete.
    Pending,
}

/// A single entry in the prefetch ring buffer.
#[derive(Clone, Debug)]
pub struct PrefetchFrame {
    /// Source this frame belongs to.
    pub source_id: SourceId,
    /// Absolute frame number within the source.
    pub frame_number: FrameNumber,
    /// Presentation timestamp of this frame.
    pub pts: TimeCode,
    /// Decoded pixel data (or `Pending`).
    pub data: PrefetchData,
}

impl PrefetchFrame {
    /// Returns `true` if the data has been filled in.
    pub fn is_ready(&self) -> bool {
        matches!(self.data, PrefetchData::Rgba(..))
    }
}

/// Aggregate statistics for the prefetch queue.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefetchStats {
    /// Number of frames with decoded data ready to consume.
    pub buffered: usize,
    /// Number of frames still waiting for decode.
    pub pending: usize,
    /// Maximum queue capacity.
    pub capacity: usize,
    /// Cache hit rate (0.0–1.0). Ratio of `get_frame` calls that
    /// returned a ready frame vs total `get_frame` calls.
    pub hit_rate: f64,
}

/// Look-ahead frame decoding queue.
///
/// Pre-decodes frames ahead of the playhead to ensure smooth playback.
/// Higher-level code should call [`request_range`](Self::request_range)
/// when the playhead moves, poll [`pending_requests`](Self::pending_requests)
/// to feed the decoder, and read decoded data via
/// [`get_frame`](Self::get_frame).
pub struct PrefetchQueue {
    /// Maximum number of frames in the ring buffer.
    buffer_size: usize,
    /// The ring buffer itself.
    buffer: VecDeque<PrefetchFrame>,
    /// Lifetime counters for hit-rate tracking.
    hits: u64,
    misses: u64,
}

impl PrefetchQueue {
    /// Create a new queue that can hold up to `buffer_size` frames.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_size` is zero.
    pub fn new(buffer_size: usize) -> Self {
        assert!(buffer_size > 0, "buffer_size must be > 0");
        Self {
            buffer_size,
            buffer: VecDeque::with_capacity(buffer_size),
            hits: 0,
            misses: 0,
        }
    }

    /// Enqueue a range of frames for prefetching.
    ///
    /// Frames that are already in the queue (matching source + frame
    /// number) are **not** duplicated. If adding the new frames would
    /// exceed `buffer_size`, the **oldest** entries are evicted.
    pub fn request_range(
        &mut self,
        source_id: SourceId,
        start: FrameNumber,
        count: usize,
    ) {
        for i in 0..count {
            let frame_number = FrameNumber(start.0 + i as u64);

            // Skip if already queued.
            let already_exists = self.buffer.iter().any(|f| {
                f.source_id == source_id && f.frame_number == frame_number
            });
            if already_exists {
                continue;
            }

            // Evict oldest if at capacity.
            if self.buffer.len() >= self.buffer_size {
                self.buffer.pop_front();
            }

            self.buffer.push_back(PrefetchFrame {
                source_id: source_id.clone(),
                frame_number,
                pts: TimeCode::ZERO, // caller can update via complete_frame
                data: PrefetchData::Pending,
            });
        }
    }

    /// Look up a prefetched frame by source and frame number.
    ///
    /// Updates internal hit/miss counters for stats.
    pub fn get_frame(
        &mut self,
        source_id: &SourceId,
        frame: FrameNumber,
    ) -> Option<&PrefetchFrame> {
        let found = self.buffer.iter().find(|f| {
            f.source_id == *source_id && f.frame_number == frame
        });
        if let Some(f) = found {
            if f.is_ready() {
                self.hits += 1;
            } else {
                self.misses += 1;
            }
            Some(f)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Immutable lookup (does **not** alter hit/miss counters).
    pub fn peek_frame(
        &self,
        source_id: &SourceId,
        frame: FrameNumber,
    ) -> Option<&PrefetchFrame> {
        self.buffer.iter().find(|f| {
            f.source_id == *source_id && f.frame_number == frame
        })
    }

    /// Mark a pending frame as decoded.
    ///
    /// Returns `true` if the frame was found and updated, `false` if
    /// there was no matching pending entry.
    pub fn complete_frame(
        &mut self,
        source_id: SourceId,
        frame: FrameNumber,
        pts: TimeCode,
        data: PrefetchData,
    ) -> bool {
        if let Some(entry) = self.buffer.iter_mut().find(|f| {
            f.source_id == source_id && f.frame_number == frame
        }) {
            entry.pts = pts;
            entry.data = data;
            true
        } else {
            false
        }
    }

    /// Clear the entire queue (e.g. after a seek).
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Clear only frames belonging to a specific source.
    pub fn clear_source(&mut self, source_id: &SourceId) {
        self.buffer.retain(|f| f.source_id != *source_id);
    }

    /// Return all `(SourceId, FrameNumber)` pairs that are still
    /// in `Pending` state and need to be decoded.
    pub fn pending_requests(&self) -> Vec<(SourceId, FrameNumber)> {
        self.buffer
            .iter()
            .filter(|f| matches!(f.data, PrefetchData::Pending))
            .map(|f| (f.source_id.clone(), f.frame_number))
            .collect()
    }

    /// Number of frames currently in the queue.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> PrefetchStats {
        let buffered = self.buffer.iter().filter(|f| f.is_ready()).count();
        let pending = self.buffer.iter().filter(|f| !f.is_ready()).count();
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        PrefetchStats {
            buffered,
            pending,
            capacity: self.buffer_size,
            hit_rate,
        }
    }

    /// Reset the hit/miss counters.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

impl std::fmt::Debug for PrefetchQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchQueue")
            .field("buffer_size", &self.buffer_size)
            .field("len", &self.buffer.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn src(name: &str) -> SourceId {
        SourceId::new(name)
    }

    fn frame(n: u64) -> FrameNumber {
        FrameNumber(n)
    }

    fn dummy_rgba(w: u32, h: u32) -> PrefetchData {
        PrefetchData::Rgba(vec![0u8; (w * h * 4) as usize], w, h)
    }

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn new_queue_is_empty() {
        let q = PrefetchQueue::new(8);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    #[should_panic(expected = "buffer_size must be > 0")]
    fn zero_buffer_panics() {
        let _ = PrefetchQueue::new(0);
    }

    // ── request_range ────────────────────────────────────────────

    #[test]
    fn request_range_creates_pending_entries() {
        let mut q = PrefetchQueue::new(16);
        q.request_range(src("a.mp4"), frame(10), 5);

        assert_eq!(q.len(), 5);
        let pending = q.pending_requests();
        assert_eq!(pending.len(), 5);
        assert_eq!(pending[0], (src("a.mp4"), frame(10)));
        assert_eq!(pending[4], (src("a.mp4"), frame(14)));
    }

    #[test]
    fn request_range_no_duplicates() {
        let mut q = PrefetchQueue::new(16);
        q.request_range(src("a.mp4"), frame(0), 5);
        q.request_range(src("a.mp4"), frame(3), 5); // overlaps 3,4

        assert_eq!(q.len(), 8); // 0..7 unique
    }

    #[test]
    fn request_range_evicts_oldest_when_full() {
        let mut q = PrefetchQueue::new(4);
        q.request_range(src("a.mp4"), frame(0), 4);
        assert_eq!(q.len(), 4);

        // Add 2 more — should evict frames 0 and 1.
        q.request_range(src("a.mp4"), frame(4), 2);
        assert_eq!(q.len(), 4);
        assert!(q.peek_frame(&src("a.mp4"), frame(0)).is_none());
        assert!(q.peek_frame(&src("a.mp4"), frame(1)).is_none());
        assert!(q.peek_frame(&src("a.mp4"), frame(4)).is_some());
        assert!(q.peek_frame(&src("a.mp4"), frame(5)).is_some());
    }

    // ── complete_frame ───────────────────────────────────────────

    #[test]
    fn complete_frame_marks_ready() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 3);

        let ok = q.complete_frame(
            src("a.mp4"),
            frame(1),
            TimeCode::from_secs(0.033),
            dummy_rgba(1920, 1080),
        );
        assert!(ok);

        let f = q.peek_frame(&src("a.mp4"), frame(1)).unwrap();
        assert!(f.is_ready());
        assert!((f.pts.as_secs() - 0.033).abs() < 1e-9);
    }

    #[test]
    fn complete_frame_returns_false_for_missing() {
        let mut q = PrefetchQueue::new(8);
        let ok = q.complete_frame(
            src("a.mp4"),
            frame(99),
            TimeCode::ZERO,
            dummy_rgba(1, 1),
        );
        assert!(!ok);
    }

    // ── get_frame & hit tracking ─────────────────────────────────

    #[test]
    fn get_frame_counts_hits_and_misses() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 2);
        q.complete_frame(src("a.mp4"), frame(0), TimeCode::ZERO, dummy_rgba(1, 1));

        // Hit: frame 0 is ready.
        let _ = q.get_frame(&src("a.mp4"), frame(0));
        // Miss: frame 1 exists but is pending.
        let _ = q.get_frame(&src("a.mp4"), frame(1));
        // Miss: frame 99 does not exist.
        let _ = q.get_frame(&src("a.mp4"), frame(99));

        let s = q.stats();
        assert_eq!(s.hit_rate, 1.0 / 3.0);
    }

    #[test]
    fn get_frame_returns_none_for_absent() {
        let mut q = PrefetchQueue::new(8);
        assert!(q.get_frame(&src("x.mp4"), frame(0)).is_none());
    }

    #[test]
    fn peek_does_not_alter_stats() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 1);
        let _ = q.peek_frame(&src("a.mp4"), frame(0));
        let s = q.stats();
        assert_eq!(s.hit_rate, 0.0);
    }

    // ── clear / clear_source ─────────────────────────────────────

    #[test]
    fn clear_empties_queue() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 5);
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn clear_source_only_removes_matching() {
        let mut q = PrefetchQueue::new(16);
        q.request_range(src("a.mp4"), frame(0), 3);
        q.request_range(src("b.mp4"), frame(0), 2);

        q.clear_source(&src("a.mp4"));
        assert_eq!(q.len(), 2);
        assert!(q.peek_frame(&src("b.mp4"), frame(0)).is_some());
        assert!(q.peek_frame(&src("a.mp4"), frame(0)).is_none());
    }

    // ── pending_requests ─────────────────────────────────────────

    #[test]
    fn pending_requests_excludes_completed() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 4);
        q.complete_frame(src("a.mp4"), frame(1), TimeCode::ZERO, dummy_rgba(1, 1));
        q.complete_frame(src("a.mp4"), frame(3), TimeCode::ZERO, dummy_rgba(1, 1));

        let pending = q.pending_requests();
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&(src("a.mp4"), frame(0))));
        assert!(pending.contains(&(src("a.mp4"), frame(2))));
    }

    // ── stats ────────────────────────────────────────────────────

    #[test]
    fn stats_capacity_matches() {
        let q = PrefetchQueue::new(16);
        assert_eq!(q.stats().capacity, 16);
    }

    #[test]
    fn stats_counts_buffered_and_pending() {
        let mut q = PrefetchQueue::new(16);
        q.request_range(src("a.mp4"), frame(0), 4);
        q.complete_frame(src("a.mp4"), frame(0), TimeCode::ZERO, dummy_rgba(1, 1));

        let s = q.stats();
        assert_eq!(s.buffered, 1);
        assert_eq!(s.pending, 3);
    }

    #[test]
    fn stats_hit_rate_no_lookups() {
        let q = PrefetchQueue::new(8);
        assert_eq!(q.stats().hit_rate, 0.0);
    }

    #[test]
    fn reset_stats_zeroes_counters() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 1);
        q.complete_frame(src("a.mp4"), frame(0), TimeCode::ZERO, dummy_rgba(1, 1));
        let _ = q.get_frame(&src("a.mp4"), frame(0));

        q.reset_stats();
        assert_eq!(q.stats().hit_rate, 0.0);
    }

    // ── multi-source ─────────────────────────────────────────────

    #[test]
    fn multiple_sources_coexist() {
        let mut q = PrefetchQueue::new(16);
        q.request_range(src("a.mp4"), frame(0), 3);
        q.request_range(src("b.mp4"), frame(100), 2);

        assert_eq!(q.len(), 5);
        assert!(q.peek_frame(&src("a.mp4"), frame(2)).is_some());
        assert!(q.peek_frame(&src("b.mp4"), frame(101)).is_some());
    }

    #[test]
    fn complete_wrong_source_returns_false() {
        let mut q = PrefetchQueue::new(8);
        q.request_range(src("a.mp4"), frame(0), 1);

        let ok = q.complete_frame(
            src("b.mp4"),
            frame(0),
            TimeCode::ZERO,
            dummy_rgba(1, 1),
        );
        assert!(!ok);
    }

    // ── Debug ────────────────────────────────────────────────────

    #[test]
    fn debug_format() {
        let q = PrefetchQueue::new(8);
        let dbg = format!("{q:?}");
        assert!(dbg.contains("PrefetchQueue"));
        assert!(dbg.contains("buffer_size: 8"));
    }

    // ── Edge: single capacity ────────────────────────────────────

    #[test]
    fn capacity_one_queue() {
        let mut q = PrefetchQueue::new(1);
        q.request_range(src("a.mp4"), frame(0), 1);
        assert_eq!(q.len(), 1);

        // Adding another evicts the first.
        q.request_range(src("a.mp4"), frame(1), 1);
        assert_eq!(q.len(), 1);
        assert!(q.peek_frame(&src("a.mp4"), frame(0)).is_none());
        assert!(q.peek_frame(&src("a.mp4"), frame(1)).is_some());
    }
}
