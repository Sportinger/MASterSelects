//! Thumbnail Generator — single-frame extraction for the timeline UI.
//!
//! The timeline in a video editor displays a strip of thumbnail images
//! for each clip so the user can visually identify content while
//! editing. This module provides an LRU-based cache for those
//! thumbnails.
//!
//! In Phase 0 the generator is a pure CPU-side cache. Later phases
//! will hook into the `DecoderPool` to trigger actual hardware decodes
//! for missing thumbnails.

use std::collections::{HashMap, VecDeque};

use ms_common::SourceId;

/// Cache key for a thumbnail — uniquely identifies a specific
/// frame at a specific resolution from a specific source.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThumbnailKey {
    /// Source media file.
    pub source_id: SourceId,
    /// Time offset in milliseconds (integer for hashing).
    pub time_ms: u64,
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
}

impl ThumbnailKey {
    pub fn new(source_id: SourceId, time_ms: u64, width: u32, height: u32) -> Self {
        Self {
            source_id,
            time_ms,
            width,
            height,
        }
    }
}

/// Decoded RGBA thumbnail image.
#[derive(Clone, Debug)]
pub struct ThumbnailData {
    /// RGBA pixel data (row-major, 4 bytes per pixel).
    pub rgba: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ThumbnailData {
    /// Create a new `ThumbnailData` from raw RGBA bytes.
    ///
    /// # Panics
    ///
    /// Panics if `rgba.len() != width * height * 4`.
    pub fn new(rgba: Vec<u8>, width: u32, height: u32) -> Self {
        assert_eq!(
            rgba.len(),
            (width as usize) * (height as usize) * 4,
            "RGBA data length must match width * height * 4"
        );
        Self { rgba, width, height }
    }

    /// Byte size of this thumbnail's pixel data.
    pub fn byte_size(&self) -> usize {
        self.rgba.len()
    }
}

/// Statistics about the thumbnail cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThumbnailStats {
    /// Number of entries currently in the cache.
    pub cached: usize,
    /// Maximum number of entries.
    pub max_cache_size: usize,
    /// Approximate total bytes used by cached pixel data.
    pub total_bytes: usize,
}

/// LRU-based thumbnail cache.
///
/// Stores decoded RGBA thumbnail images keyed by
/// `(SourceId, time_ms, width, height)`. When the cache reaches its
/// capacity, the **least-recently-inserted** entry is evicted.
pub struct ThumbnailGenerator {
    /// The actual cache data.
    cache: HashMap<ThumbnailKey, ThumbnailData>,
    /// Insertion order for LRU eviction.
    order: VecDeque<ThumbnailKey>,
    /// Maximum number of entries.
    max_cache_size: usize,
}

impl ThumbnailGenerator {
    /// Create a new generator with the given cache capacity.
    ///
    /// # Panics
    ///
    /// Panics if `max_cache_size` is zero.
    pub fn new(max_cache_size: usize) -> Self {
        assert!(max_cache_size > 0, "max_cache_size must be > 0");
        Self {
            cache: HashMap::new(),
            order: VecDeque::new(),
            max_cache_size,
        }
    }

    /// Look up a cached thumbnail.
    pub fn get_thumbnail(&self, key: &ThumbnailKey) -> Option<&ThumbnailData> {
        self.cache.get(key)
    }

    /// Store a thumbnail in the cache.
    ///
    /// If a thumbnail with the same key already exists it is replaced
    /// (and keeps its position in the LRU order). If the cache is at
    /// capacity the oldest entry is evicted first.
    pub fn store_thumbnail(&mut self, key: ThumbnailKey, data: ThumbnailData) {
        // If the key already exists, just replace the data in-place
        // without touching the LRU order.
        if let Some(existing) = self.cache.get_mut(&key) {
            *existing = data;
            return;
        }

        self.evict_if_full();
        self.order.push_back(key.clone());
        self.cache.insert(key, data);
    }

    /// Check whether a thumbnail is already cached.
    pub fn has_thumbnail(&self, key: &ThumbnailKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Remove all entries from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    /// Remove all entries belonging to a specific source.
    pub fn clear_source(&mut self, source_id: &SourceId) {
        self.order.retain(|k| k.source_id != *source_id);
        self.cache.retain(|k, _| k.source_id != *source_id);
    }

    /// Evict the oldest entry if the cache is full.
    pub fn evict_if_full(&mut self) {
        while self.cache.len() >= self.max_cache_size {
            if let Some(old_key) = self.order.pop_front() {
                self.cache.remove(&old_key);
            } else {
                break;
            }
        }
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Cache statistics.
    pub fn stats(&self) -> ThumbnailStats {
        let total_bytes: usize = self.cache.values().map(|d| d.byte_size()).sum();
        ThumbnailStats {
            cached: self.cache.len(),
            max_cache_size: self.max_cache_size,
            total_bytes,
        }
    }
}

impl std::fmt::Debug for ThumbnailGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThumbnailGenerator")
            .field("cached", &self.cache.len())
            .field("max_cache_size", &self.max_cache_size)
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

    fn key(name: &str, time_ms: u64, w: u32, h: u32) -> ThumbnailKey {
        ThumbnailKey::new(src(name), time_ms, w, h)
    }

    fn dummy_data(w: u32, h: u32) -> ThumbnailData {
        ThumbnailData::new(vec![128u8; (w * h * 4) as usize], w, h)
    }

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn new_cache_is_empty() {
        let gen = ThumbnailGenerator::new(16);
        assert!(gen.is_empty());
        assert_eq!(gen.len(), 0);
    }

    #[test]
    #[should_panic(expected = "max_cache_size must be > 0")]
    fn zero_capacity_panics() {
        let _ = ThumbnailGenerator::new(0);
    }

    // ── Store & Get ──────────────────────────────────────────────

    #[test]
    fn store_and_retrieve() {
        let mut gen = ThumbnailGenerator::new(16);
        let k = key("a.mp4", 1000, 160, 90);
        gen.store_thumbnail(k.clone(), dummy_data(160, 90));

        assert!(gen.has_thumbnail(&k));
        let thumb = gen.get_thumbnail(&k).unwrap();
        assert_eq!(thumb.width, 160);
        assert_eq!(thumb.height, 90);
        assert_eq!(thumb.byte_size(), 160 * 90 * 4);
    }

    #[test]
    fn get_missing_returns_none() {
        let gen = ThumbnailGenerator::new(16);
        assert!(gen.get_thumbnail(&key("x.mp4", 0, 1, 1)).is_none());
    }

    #[test]
    fn store_replaces_existing() {
        let mut gen = ThumbnailGenerator::new(16);
        let k = key("a.mp4", 0, 2, 2);
        gen.store_thumbnail(k.clone(), ThumbnailData::new(vec![0; 16], 2, 2));
        gen.store_thumbnail(k.clone(), ThumbnailData::new(vec![255; 16], 2, 2));

        let thumb = gen.get_thumbnail(&k).unwrap();
        assert_eq!(thumb.rgba[0], 255);
        assert_eq!(gen.len(), 1, "should not duplicate");
    }

    // ── Eviction ─────────────────────────────────────────────────

    #[test]
    fn evicts_oldest_when_full() {
        let mut gen = ThumbnailGenerator::new(2);
        gen.store_thumbnail(key("a.mp4", 0, 1, 1), dummy_data(1, 1));
        gen.store_thumbnail(key("b.mp4", 0, 1, 1), dummy_data(1, 1));
        // Full now. Storing one more should evict "a".
        gen.store_thumbnail(key("c.mp4", 0, 1, 1), dummy_data(1, 1));

        assert_eq!(gen.len(), 2);
        assert!(!gen.has_thumbnail(&key("a.mp4", 0, 1, 1)));
        assert!(gen.has_thumbnail(&key("b.mp4", 0, 1, 1)));
        assert!(gen.has_thumbnail(&key("c.mp4", 0, 1, 1)));
    }

    #[test]
    fn evict_if_full_is_noop_when_not_full() {
        let mut gen = ThumbnailGenerator::new(8);
        gen.store_thumbnail(key("a.mp4", 0, 1, 1), dummy_data(1, 1));
        gen.evict_if_full();
        assert_eq!(gen.len(), 1);
    }

    // ── Clear ────────────────────────────────────────────────────

    #[test]
    fn clear_removes_all() {
        let mut gen = ThumbnailGenerator::new(16);
        gen.store_thumbnail(key("a.mp4", 0, 1, 1), dummy_data(1, 1));
        gen.store_thumbnail(key("b.mp4", 0, 1, 1), dummy_data(1, 1));
        gen.clear();
        assert!(gen.is_empty());
    }

    #[test]
    fn clear_source_only_removes_matching() {
        let mut gen = ThumbnailGenerator::new(16);
        gen.store_thumbnail(key("a.mp4", 0, 1, 1), dummy_data(1, 1));
        gen.store_thumbnail(key("a.mp4", 1000, 1, 1), dummy_data(1, 1));
        gen.store_thumbnail(key("b.mp4", 0, 1, 1), dummy_data(1, 1));

        gen.clear_source(&src("a.mp4"));
        assert_eq!(gen.len(), 1);
        assert!(gen.has_thumbnail(&key("b.mp4", 0, 1, 1)));
    }

    // ── Stats ────────────────────────────────────────────────────

    #[test]
    fn stats_empty_cache() {
        let gen = ThumbnailGenerator::new(8);
        let s = gen.stats();
        assert_eq!(
            s,
            ThumbnailStats {
                cached: 0,
                max_cache_size: 8,
                total_bytes: 0,
            }
        );
    }

    #[test]
    fn stats_counts_bytes() {
        let mut gen = ThumbnailGenerator::new(16);
        gen.store_thumbnail(key("a.mp4", 0, 10, 10), dummy_data(10, 10)); // 10*10*4 = 400
        gen.store_thumbnail(key("b.mp4", 0, 5, 5), dummy_data(5, 5)); // 5*5*4 = 100

        let s = gen.stats();
        assert_eq!(s.cached, 2);
        assert_eq!(s.total_bytes, 500);
    }

    // ── ThumbnailData ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "RGBA data length must match")]
    fn thumbnail_data_wrong_size_panics() {
        ThumbnailData::new(vec![0; 10], 100, 100);
    }

    #[test]
    fn thumbnail_data_byte_size() {
        let d = dummy_data(320, 180);
        assert_eq!(d.byte_size(), 320 * 180 * 4);
    }

    // ── ThumbnailKey ─────────────────────────────────────────────

    #[test]
    fn key_equality() {
        let k1 = key("a.mp4", 1000, 160, 90);
        let k2 = key("a.mp4", 1000, 160, 90);
        let k3 = key("a.mp4", 2000, 160, 90);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn key_different_resolution() {
        let k1 = key("a.mp4", 1000, 160, 90);
        let k2 = key("a.mp4", 1000, 320, 180);
        assert_ne!(k1, k2);
    }

    // ── Debug ────────────────────────────────────────────────────

    #[test]
    fn debug_format() {
        let gen = ThumbnailGenerator::new(8);
        let dbg = format!("{gen:?}");
        assert!(dbg.contains("ThumbnailGenerator"));
        assert!(dbg.contains("max_cache_size: 8"));
    }

    // ── Capacity one ─────────────────────────────────────────────

    #[test]
    fn cache_capacity_one() {
        let mut gen = ThumbnailGenerator::new(1);
        gen.store_thumbnail(key("a.mp4", 0, 1, 1), dummy_data(1, 1));
        gen.store_thumbnail(key("b.mp4", 0, 1, 1), dummy_data(1, 1));
        assert_eq!(gen.len(), 1);
        assert!(gen.has_thumbnail(&key("b.mp4", 0, 1, 1)));
        assert!(!gen.has_thumbnail(&key("a.mp4", 0, 1, 1)));
    }

    // ── Multiple timestamps same source ──────────────────────────

    #[test]
    fn multiple_timestamps() {
        let mut gen = ThumbnailGenerator::new(16);
        for t in 0..10 {
            gen.store_thumbnail(
                key("a.mp4", t * 1000, 160, 90),
                dummy_data(160, 90),
            );
        }
        assert_eq!(gen.len(), 10);
        assert!(gen.has_thumbnail(&key("a.mp4", 0, 160, 90)));
        assert!(gen.has_thumbnail(&key("a.mp4", 9000, 160, 90)));
    }
}
