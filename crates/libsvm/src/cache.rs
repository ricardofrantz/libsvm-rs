//! LRU kernel cache matching the original LIBSVM.
//!
//! The cache stores rows of the kernel matrix Q as `Qfloat` (`f32`) slices.
//! When memory is exhausted, the least-recently-used row is evicted.
//!
//! The C++ original uses a doubly-linked circular list with raw pointers.
//! This Rust version uses a `VecDeque` for the LRU order and `Vec<Option<Vec<Qfloat>>>`
//! for per-row data, avoiding unsafe code while matching the semantics.

/// Element type for cached kernel matrix rows. Matches LIBSVM's `Qfloat = float`.
pub type Qfloat = f32;

/// LRU cache for kernel matrix rows.
///
/// Each of the `l` data items may have a cached row of length up to `l`.
/// The cache tracks how much memory (in Qfloat units) is in use and evicts
/// LRU entries when the budget is exceeded.
pub struct Cache {
    /// Number of data items (rows in the Q matrix).
    l: usize,
    /// Available budget in Qfloat units.
    size: usize,
    /// Per-row cached data. `None` means not cached.
    data: Vec<Option<Vec<Qfloat>>>,
    /// Per-row cached length (how many elements are valid).
    len: Vec<usize>,
    /// LRU order: front = least recently used, back = most recently used.
    /// Contains indices of rows currently in the cache.
    lru: Vec<usize>,
}

impl Cache {
    /// Create a new cache for `l` data items with `size_bytes` of memory.
    pub fn new(l: usize, size_bytes: usize) -> Self {
        // Convert bytes to Qfloat units
        let mut size = size_bytes / std::mem::size_of::<Qfloat>();
        // Subtract header overhead (metadata per row)
        let header_size = l * 3 * std::mem::size_of::<usize>() / std::mem::size_of::<Qfloat>();
        // Cache must be large enough for at least two columns
        size = size.max(2 * l + header_size).saturating_sub(header_size);

        Cache {
            l,
            size,
            data: (0..l).map(|_| None).collect(),
            len: vec![0; l],
            lru: Vec::new(),
        }
    }

    /// Request data for row `index` of length `request_len`.
    ///
    /// Returns `(data, start)` where `data` is the cached row slice and
    /// `start` is the position from which data needs to be filled.
    /// If `start >= request_len`, the entire row was already cached.
    ///
    /// The caller must fill `data[start..request_len]` with kernel values.
    pub fn get_data(&mut self, index: usize, request_len: usize) -> (&mut [Qfloat], usize) {
        assert!(index < self.l);

        // Remove from LRU if present
        if self.len[index] > 0 {
            self.lru_remove(index);
        }

        let old_len = self.len[index];
        let more = request_len.saturating_sub(old_len);

        if more > 0 {
            // Evict until we have enough space
            while self.size < more {
                if let Some(victim) = self.lru.first().copied() {
                    self.lru.remove(0);
                    let victim_len = self.len[victim];
                    self.size += victim_len;
                    self.data[victim] = None;
                    self.len[victim] = 0;
                } else {
                    break;
                }
            }

            // Allocate or extend
            let entry = self.data[index].get_or_insert_with(Vec::new);
            entry.resize(request_len, 0.0);
            self.size -= more;
            self.len[index] = request_len;
        }

        // Insert at back of LRU (most recently used)
        self.lru.push(index);

        let start = old_len;
        (self.data[index].as_mut().unwrap().as_mut_slice(), start)
    }

    /// Swap indices `i` and `j` in the cache.
    ///
    /// Used by the solver when rearranging the working set.
    pub fn swap_index(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }

        // Remove both from LRU if present
        if self.len[i] > 0 {
            self.lru_remove(i);
        }
        if self.len[j] > 0 {
            self.lru_remove(j);
        }

        // Swap data and lengths
        self.data.swap(i, j);
        self.len.swap(i, j);

        // Re-insert into LRU if they have data
        if self.len[i] > 0 {
            self.lru.push(i);
        }
        if self.len[j] > 0 {
            self.lru.push(j);
        }
    }

    fn lru_remove(&mut self, index: usize) {
        if let Some(pos) = self.lru.iter().position(|&x| x == index) {
            self.lru.remove(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_get_and_fill() {
        // Cache for 3 items, 100 bytes
        let mut cache = Cache::new(3, 100);
        let (data, start) = cache.get_data(0, 3);
        assert_eq!(start, 0); // nothing cached yet
        assert_eq!(data.len(), 3);
        data[0] = 1.0;
        data[1] = 2.0;
        data[2] = 3.0;

        // Second access should return start=3 (fully cached)
        let (data, start) = cache.get_data(0, 3);
        assert_eq!(start, 3);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
    }

    #[test]
    fn extend_cached_row() {
        let mut cache = Cache::new(3, 1000);
        let (data, start) = cache.get_data(0, 2);
        assert_eq!(start, 0);
        data[0] = 10.0;
        data[1] = 20.0;

        // Request longer row
        let (data, start) = cache.get_data(0, 3);
        assert_eq!(start, 2); // only need to fill [2..3)
        assert_eq!(data[0], 10.0);
        assert_eq!(data[1], 20.0);
        data[2] = 30.0;
    }

    #[test]
    fn lru_eviction() {
        // Use 10 items to make the minimum cache size meaningful.
        // Min size = 2*l = 20 Qfloats. Request rows of 10, so only 2 fit.
        let l = 10;
        // Give just enough bytes for ~20 Qfloats + header overhead
        let bytes = (2 * l + l * 3) * std::mem::size_of::<Qfloat>();
        let mut cache = Cache::new(l, bytes);

        // Fill row 0 with l elements
        let (data, start) = cache.get_data(0, l);
        assert_eq!(start, 0);
        data[0] = 1.0;

        // Fill row 1 with l elements
        let (data, start) = cache.get_data(1, l);
        assert_eq!(start, 0);
        data[0] = 3.0;

        // Fill row 2 — should evict row 0 (LRU)
        let (data, start) = cache.get_data(2, l);
        assert_eq!(start, 0);
        data[0] = 5.0;

        // Row 0 should have been evicted
        let (_, start) = cache.get_data(0, l);
        assert_eq!(start, 0); // needs to be re-filled
    }

    #[test]
    fn swap_index_works() {
        let mut cache = Cache::new(3, 1000);
        let (data, _) = cache.get_data(0, 2);
        data[0] = 10.0;
        data[1] = 20.0;

        cache.swap_index(0, 2);

        // Row 2 should now have the data from row 0
        let (data, start) = cache.get_data(2, 2);
        assert_eq!(start, 2); // already cached
        assert_eq!(data[0], 10.0);
        assert_eq!(data[1], 20.0);

        // Row 0 should be empty
        let (_, start) = cache.get_data(0, 2);
        assert_eq!(start, 0);
    }
}
