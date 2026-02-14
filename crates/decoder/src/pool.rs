//! Decoder Pool — manages multiple hardware decoder instances.
//!
//! The `DecoderPool` allocates one decoder per active video source.
//! When playback begins, a decoder is created (or reused from the pool)
//! for each visible clip. When a clip scrolls off-screen or playback
//! stops, its decoder is returned to the idle set. If the pool reaches
//! its capacity limit, the least-recently-used idle decoder is evicted.
//!
//! This design keeps GPU resources bounded while minimising the cost of
//! re-creating decoder sessions during scrubbing or multi-track editing.

use std::collections::HashMap;
use std::time::Instant;

use ms_common::{DecodeError, DecoderConfig, SourceId, VideoCodec};

/// Statistics about the current state of the pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolStats {
    /// Number of decoders currently in active use.
    pub active: usize,
    /// Number of decoders sitting idle (available for reuse).
    pub available: usize,
    /// Hard maximum number of decoders the pool will maintain.
    pub max: usize,
    /// Total frames decoded across all slots since pool creation.
    pub total_frames_decoded: u64,
}

/// A single slot in the decoder pool.
///
/// In Phase 0 this acts as a bookkeeping record. Once real GPU
/// decoder sessions are wired in, the slot will own the `HwDecoder`
/// handle and the associated VRAM resources.
#[derive(Debug)]
pub struct DecoderSlot {
    /// Source this slot is bound to.
    pub source_id: SourceId,
    /// Configuration used to create the decoder.
    pub config: DecoderConfig,
    /// Monotonic timestamp of the last time this slot was accessed.
    pub last_used: Instant,
    /// Running counter of frames decoded through this slot.
    pub frames_decoded: u64,
}

/// Manages a bounded pool of hardware decoder slots.
///
/// # Thread safety
///
/// The pool itself is `Send` but **not** `Sync`. Higher-level code
/// (e.g. the render thread) should wrap it in `parking_lot::Mutex` if
/// shared access is required.
pub struct DecoderPool {
    /// Active decoders keyed by source.
    active: HashMap<SourceId, DecoderSlot>,
    /// Hard limit on the number of active decoders.
    max_decoders: usize,
}

impl DecoderPool {
    /// Create a new pool with the given maximum capacity.
    ///
    /// # Panics
    ///
    /// Panics if `max_decoders` is zero.
    pub fn new(max_decoders: usize) -> Self {
        assert!(max_decoders > 0, "max_decoders must be > 0");
        Self {
            active: HashMap::new(),
            max_decoders,
        }
    }

    /// Get or create a decoder slot for a specific source.
    ///
    /// If a slot for `source_id` already exists it is returned directly.
    /// Otherwise a new slot is allocated. If the pool is at capacity the
    /// **least-recently-used** slot is evicted first.
    pub fn get_decoder(
        &mut self,
        source_id: &SourceId,
        config: &DecoderConfig,
    ) -> Result<&mut DecoderSlot, DecodeError> {
        // Fast path: source already has a slot.
        if self.active.contains_key(source_id) {
            let slot = self.active.get_mut(source_id).expect("just checked");
            slot.last_used = Instant::now();
            return Ok(slot);
        }

        // Evict LRU if at capacity.
        if self.active.len() >= self.max_decoders {
            self.evict_lru();
        }

        // Create a new slot.
        let slot = DecoderSlot {
            source_id: source_id.clone(),
            config: config.clone(),
            last_used: Instant::now(),
            frames_decoded: 0,
        };
        self.active.insert(source_id.clone(), slot);
        Ok(self.active.get_mut(source_id).expect("just inserted"))
    }

    /// Explicitly release a decoder slot back to the pool (removes it).
    pub fn release(&mut self, source_id: &SourceId) {
        self.active.remove(source_id);
    }

    /// Evict all slots that have been idle longer than `max_idle_secs`.
    pub fn evict_idle(&mut self, max_idle_secs: f64) {
        let now = Instant::now();
        self.active.retain(|_id, slot| {
            let idle = now.duration_since(slot.last_used).as_secs_f64();
            idle < max_idle_secs
        });
    }

    /// Return the number of currently active slots.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Return the configured maximum capacity.
    pub fn max_capacity(&self) -> usize {
        self.max_decoders
    }

    /// Return current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let total_frames: u64 = self.active.values().map(|s| s.frames_decoded).sum();
        PoolStats {
            active: self.active.len(),
            available: self.max_decoders.saturating_sub(self.active.len()),
            max: self.max_decoders,
            total_frames_decoded: total_frames,
        }
    }

    /// Check whether a decoder exists for the given source.
    pub fn contains(&self, source_id: &SourceId) -> bool {
        self.active.contains_key(source_id)
    }

    /// Immutable access to a slot (e.g. for stats display).
    pub fn get_slot(&self, source_id: &SourceId) -> Option<&DecoderSlot> {
        self.active.get(source_id)
    }

    /// Return the codecs of all active decoders.
    pub fn active_codecs(&self) -> Vec<VideoCodec> {
        self.active.values().map(|s| s.config.codec).collect()
    }

    /// Remove **all** decoders from the pool.
    pub fn clear(&mut self) {
        self.active.clear();
    }

    // ── internal helpers ──────────────────────────────────────────

    /// Evict the single least-recently-used slot.
    fn evict_lru(&mut self) {
        if let Some((lru_id, _)) = self
            .active
            .iter()
            .min_by_key(|(_id, slot)| slot.last_used)
        {
            let lru_id = lru_id.clone();
            self.active.remove(&lru_id);
        }
    }
}

impl std::fmt::Debug for DecoderPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderPool")
            .field("active", &self.active.len())
            .field("max_decoders", &self.max_decoders)
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ms_common::{Resolution, VideoCodec};
    use std::thread;
    use std::time::Duration;

    fn make_config(codec: VideoCodec) -> DecoderConfig {
        DecoderConfig::new(codec, Resolution::HD)
    }

    fn src(name: &str) -> SourceId {
        SourceId::new(name)
    }

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn new_pool_is_empty() {
        let pool = DecoderPool::new(4);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.max_capacity(), 4);
    }

    #[test]
    #[should_panic(expected = "max_decoders must be > 0")]
    fn zero_capacity_panics() {
        let _ = DecoderPool::new(0);
    }

    // ── Get / Create ─────────────────────────────────────────────

    #[test]
    fn get_creates_new_slot() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);
        let slot = pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        assert_eq!(slot.source_id, src("a.mp4"));
        assert_eq!(slot.frames_decoded, 0);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn get_returns_existing_slot() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap().frames_decoded = 42;

        let slot = pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        assert_eq!(slot.frames_decoded, 42, "should return the same slot");
    }

    #[test]
    fn get_updates_last_used() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        let t1 = pool.get_slot(&src("a.mp4")).unwrap().last_used;

        thread::sleep(Duration::from_millis(5));
        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        let t2 = pool.get_slot(&src("a.mp4")).unwrap().last_used;

        assert!(t2 > t1, "last_used should advance on re-access");
    }

    #[test]
    fn multiple_sources() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();
        pool.get_decoder(&src("c.mp4"), &cfg).unwrap();

        assert_eq!(pool.active_count(), 3);
        assert!(pool.contains(&src("a.mp4")));
        assert!(pool.contains(&src("b.mp4")));
        assert!(pool.contains(&src("c.mp4")));
    }

    // ── LRU Eviction ─────────────────────────────────────────────

    #[test]
    fn lru_eviction_when_full() {
        let mut pool = DecoderPool::new(2);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        thread::sleep(Duration::from_millis(5));
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();
        thread::sleep(Duration::from_millis(5));

        // Pool is full (2/2). Adding "c" should evict "a" (oldest).
        pool.get_decoder(&src("c.mp4"), &cfg).unwrap();

        assert_eq!(pool.active_count(), 2);
        assert!(!pool.contains(&src("a.mp4")), "LRU 'a' should be evicted");
        assert!(pool.contains(&src("b.mp4")));
        assert!(pool.contains(&src("c.mp4")));
    }

    #[test]
    fn lru_evicts_correct_slot_after_reaccess() {
        let mut pool = DecoderPool::new(2);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        thread::sleep(Duration::from_millis(5));
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();
        thread::sleep(Duration::from_millis(5));

        // Re-access "a" so it becomes the most-recently-used.
        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        thread::sleep(Duration::from_millis(5));

        // "b" is now LRU. Adding "c" should evict "b".
        pool.get_decoder(&src("c.mp4"), &cfg).unwrap();

        assert!(!pool.contains(&src("b.mp4")), "LRU 'b' should be evicted");
        assert!(pool.contains(&src("a.mp4")));
        assert!(pool.contains(&src("c.mp4")));
    }

    // ── Release ──────────────────────────────────────────────────

    #[test]
    fn release_removes_slot() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        assert_eq!(pool.active_count(), 1);

        pool.release(&src("a.mp4"));
        assert_eq!(pool.active_count(), 0);
        assert!(!pool.contains(&src("a.mp4")));
    }

    #[test]
    fn release_nonexistent_is_noop() {
        let mut pool = DecoderPool::new(4);
        pool.release(&src("nonexistent.mp4"));
        assert_eq!(pool.active_count(), 0);
    }

    // ── Evict idle ───────────────────────────────────────────────

    #[test]
    fn evict_idle_removes_old_slots() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("old.mp4"), &cfg).unwrap();
        thread::sleep(Duration::from_millis(60));
        pool.get_decoder(&src("new.mp4"), &cfg).unwrap();

        // Evict anything idle > 30ms.
        pool.evict_idle(0.030);

        assert!(!pool.contains(&src("old.mp4")), "old should be evicted");
        assert!(pool.contains(&src("new.mp4")), "new should survive");
    }

    #[test]
    fn evict_idle_keeps_all_when_threshold_high() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();

        pool.evict_idle(9999.0);
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn evict_idle_on_empty_pool_is_noop() {
        let mut pool = DecoderPool::new(4);
        pool.evict_idle(0.0);
        assert_eq!(pool.active_count(), 0);
    }

    // ── Clear ────────────────────────────────────────────────────

    #[test]
    fn clear_removes_everything() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();

        pool.clear();
        assert_eq!(pool.active_count(), 0);
    }

    // ── Stats ────────────────────────────────────────────────────

    #[test]
    fn stats_empty_pool() {
        let pool = DecoderPool::new(4);
        let s = pool.stats();
        assert_eq!(
            s,
            PoolStats {
                active: 0,
                available: 4,
                max: 4,
                total_frames_decoded: 0,
            }
        );
    }

    #[test]
    fn stats_after_decode_activity() {
        let mut pool = DecoderPool::new(4);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg)
            .unwrap()
            .frames_decoded = 100;
        pool.get_decoder(&src("b.mp4"), &cfg)
            .unwrap()
            .frames_decoded = 50;

        let s = pool.stats();
        assert_eq!(s.active, 2);
        assert_eq!(s.available, 2);
        assert_eq!(s.total_frames_decoded, 150);
    }

    // ── Active codecs ────────────────────────────────────────────

    #[test]
    fn active_codecs_list() {
        let mut pool = DecoderPool::new(4);

        pool.get_decoder(
            &src("a.mp4"),
            &make_config(VideoCodec::H264),
        )
        .unwrap();
        pool.get_decoder(
            &src("b.mp4"),
            &make_config(VideoCodec::H265),
        )
        .unwrap();

        let mut codecs = pool.active_codecs();
        codecs.sort_by_key(|c| format!("{c:?}"));
        assert_eq!(codecs.len(), 2);
        assert!(codecs.contains(&VideoCodec::H264));
        assert!(codecs.contains(&VideoCodec::H265));
    }

    // ── Debug display ────────────────────────────────────────────

    #[test]
    fn debug_format() {
        let pool = DecoderPool::new(4);
        let dbg = format!("{pool:?}");
        assert!(dbg.contains("DecoderPool"));
        assert!(dbg.contains("max_decoders: 4"));
    }

    // ── Capacity edge cases ──────────────────────────────────────

    #[test]
    fn pool_capacity_one() {
        let mut pool = DecoderPool::new(1);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();

        assert_eq!(pool.active_count(), 1);
        assert!(pool.contains(&src("b.mp4")));
        assert!(!pool.contains(&src("a.mp4")));
    }

    #[test]
    fn release_frees_capacity_for_new_slot() {
        let mut pool = DecoderPool::new(2);
        let cfg = make_config(VideoCodec::H264);

        pool.get_decoder(&src("a.mp4"), &cfg).unwrap();
        pool.get_decoder(&src("b.mp4"), &cfg).unwrap();
        assert_eq!(pool.active_count(), 2);

        pool.release(&src("a.mp4"));
        assert_eq!(pool.active_count(), 1);

        // Now there is room — no eviction needed.
        pool.get_decoder(&src("c.mp4"), &cfg).unwrap();
        assert_eq!(pool.active_count(), 2);
        assert!(pool.contains(&src("b.mp4")));
        assert!(pool.contains(&src("c.mp4")));
    }
}
