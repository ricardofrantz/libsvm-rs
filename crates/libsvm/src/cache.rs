//! LRU kernel cache matching the original LIBSVM.
//!
//! The cache stores rows of the kernel matrix Q as `Qfloat` (`f32`) slices.
//! When memory is exhausted, the least-recently-used row is evicted.
//!
//! The C++ original uses a doubly-linked circular list with raw pointers.
//! This Rust version uses an index-based circular doubly-linked list for
//! O(1) LRU operations, avoiding unsafe code while matching the semantics.

/// Element type for cached kernel matrix rows. Matches LIBSVM's `Qfloat = float`.
pub type Qfloat = f32;

/// Sentinel index representing "no link" in the LRU list.
const NONE: usize = usize::MAX;

/// Per-row LRU node: stores prev/next indices for a circular doubly-linked list.
struct LruNode {
    prev: usize,
    next: usize,
}

/// LRU cache for kernel matrix rows.
///
/// Each of the `l` data items may have a cached row of length up to `l`.
/// The cache tracks how much memory (in Qfloat units) is in use and evicts
/// LRU entries when the budget is exceeded.
///
/// The LRU list is a circular doubly-linked list using array indices.
/// Index `l` is the sentinel head node. All operations (insert, remove,
/// evict) are O(1).
pub struct Cache {
    /// Number of data items (rows in the Q matrix).
    l: usize,
    /// Available budget in Qfloat units.
    size: usize,
    /// Per-row cached data. `None` means not cached.
    data: Vec<Option<Vec<Qfloat>>>,
    /// Per-row cached length (how many elements are valid).
    len: Vec<usize>,
    /// LRU doubly-linked list nodes. Index `l` is the sentinel head.
    /// Nodes with `prev == NONE` are not in the LRU list.
    nodes: Vec<LruNode>,
}

impl Cache {
    /// Create a new cache for `l` data items with `size_bytes` of memory.
    pub fn new(l: usize, size_bytes: usize) -> Self {
        // Convert bytes to Qfloat units
        let mut size = size_bytes / std::mem::size_of::<Qfloat>();
        // Subtract header overhead (metadata per row)
        let header_size = l * std::mem::size_of::<LruNode>() / std::mem::size_of::<Qfloat>();
        // Cache must be large enough for at least two columns
        size = size.max(2 * l + header_size).saturating_sub(header_size);

        // Create l+1 nodes: 0..l for data rows, l for sentinel head
        let mut nodes: Vec<LruNode> = (0..l)
            .map(|_| LruNode { prev: NONE, next: NONE })
            .collect();
        // Sentinel head points to itself (empty list)
        nodes.push(LruNode { prev: l, next: l });

        Cache {
            l,
            size,
            data: (0..l).map(|_| None).collect(),
            len: vec![0; l],
            nodes,
        }
    }

    /// Remove node `i` from the LRU list. O(1).
    #[inline]
    fn lru_delete(&mut self, i: usize) {
        let prev = self.nodes[i].prev;
        let next = self.nodes[i].next;
        self.nodes[prev].next = next;
        self.nodes[next].prev = prev;
        self.nodes[i].prev = NONE;
        self.nodes[i].next = NONE;
    }

    /// Insert node `i` at the back of the LRU list (most recently used). O(1).
    #[inline]
    fn lru_insert(&mut self, i: usize) {
        let head = self.l; // sentinel
        let tail = self.nodes[head].prev;
        self.nodes[i].next = head;
        self.nodes[i].prev = tail;
        self.nodes[tail].next = i;
        self.nodes[head].prev = i;
    }

    /// Check if node `i` is in the LRU list.
    #[inline]
    fn in_lru(&self, i: usize) -> bool {
        self.nodes[i].prev != NONE
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

        // Remove from LRU if present (will re-insert at tail)
        if self.in_lru(index) {
            self.lru_delete(index);
        }

        let old_len = self.len[index];
        let more = request_len.saturating_sub(old_len);

        if more > 0 {
            // Evict LRU entries until we have enough space
            let head = self.l;
            while self.size < more {
                let victim = self.nodes[head].next;
                if victim == head {
                    break; // list empty
                }
                self.lru_delete(victim);
                self.size += self.len[victim];
                self.data[victim] = None;
                self.len[victim] = 0;
            }

            // Allocate or extend
            let entry = self.data[index].get_or_insert_with(Vec::new);
            entry.resize(request_len, 0.0);
            self.size -= more;
            self.len[index] = request_len;
        }

        // Insert at back of LRU (most recently used)
        self.lru_insert(index);

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

        let i_in = self.in_lru(i);
        let j_in = self.in_lru(j);

        if i_in {
            self.lru_delete(i);
        }
        if j_in {
            self.lru_delete(j);
        }

        // Swap data and lengths
        self.data.swap(i, j);
        self.len.swap(i, j);

        // Re-insert with swapped identities
        if i_in {
            self.lru_insert(j);
        }
        if j_in {
            self.lru_insert(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_get_and_fill() {
        let mut cache = Cache::new(3, 100);
        let (data, start) = cache.get_data(0, 3);
        assert_eq!(start, 0);
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

        let (data, start) = cache.get_data(0, 3);
        assert_eq!(start, 2);
        assert_eq!(data[0], 10.0);
        assert_eq!(data[1], 20.0);
        data[2] = 30.0;
    }

    #[test]
    fn lru_eviction() {
        let l = 10;
        let bytes = (2 * l + l * 3) * std::mem::size_of::<Qfloat>();
        let mut cache = Cache::new(l, bytes);

        let (data, start) = cache.get_data(0, l);
        assert_eq!(start, 0);
        data[0] = 1.0;

        let (data, start) = cache.get_data(1, l);
        assert_eq!(start, 0);
        data[0] = 3.0;

        // Should evict row 0 (LRU)
        let (data, start) = cache.get_data(2, l);
        assert_eq!(start, 0);
        data[0] = 5.0;

        // Row 0 evicted
        let (_, start) = cache.get_data(0, l);
        assert_eq!(start, 0);
    }

    #[test]
    fn lru_order_respects_access() {
        // Verify that re-accessing a row moves it to MRU position.
        // l=5, row_len=5: min cache = 2*5=10 Qfloats. Budget = 3 rows = 15.
        let l = 5;
        let row_len = l;
        let header = l * std::mem::size_of::<LruNode>() / std::mem::size_of::<Qfloat>();
        let budget = 3 * row_len + header;
        let bytes = budget * std::mem::size_of::<Qfloat>();
        let mut cache = Cache::new(l, bytes);

        // Fill rows 0, 1, 2. LRU order: 0(oldest), 1, 2(newest)
        let (d, _) = cache.get_data(0, row_len);
        d[0] = 10.0;
        let (d, _) = cache.get_data(1, row_len);
        d[0] = 20.0;
        let (d, _) = cache.get_data(2, row_len);
        d[0] = 30.0;

        // Touch row 0 → LRU order: 1(oldest), 2, 0(newest)
        let (d, start) = cache.get_data(0, row_len);
        assert_eq!(start, row_len); // already cached, no fill needed
        assert_eq!(d[0], 10.0);

        // Insert row 3 → must evict. Row 1 is LRU, so it gets evicted.
        // LRU order: 2, 0, 3(newest)
        let (d, start) = cache.get_data(3, row_len);
        assert_eq!(start, 0); // new row, needs fill
        d[0] = 40.0;

        // Row 1 was evicted
        assert!(cache.data[1].is_none());
        // Row 0 and 2 are still cached
        assert!(cache.data[0].is_some());
        assert!(cache.data[2].is_some());
    }

    #[test]
    fn swap_index_works() {
        let mut cache = Cache::new(3, 1000);
        let (data, _) = cache.get_data(0, 2);
        data[0] = 10.0;
        data[1] = 20.0;

        cache.swap_index(0, 2);

        let (data, start) = cache.get_data(2, 2);
        assert_eq!(start, 2);
        assert_eq!(data[0], 10.0);
        assert_eq!(data[1], 20.0);

        let (_, start) = cache.get_data(0, 2);
        assert_eq!(start, 0);
    }
}
