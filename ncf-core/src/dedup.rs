use crate::schema::ChunkRef;
use std::collections::HashMap;

/// Metadata used by the deduplication stage of NCF writing.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// The chunk reference metadata.
    pub chunk_ref: ChunkRef,
    /// If this chunk was deduplicated, the canonical offset in the file.
    pub canonical_offset: Option<u64>,
}

/// Deduplication cache for chunk payloads.
pub struct DedupCache {
    lookup: HashMap<[u8; 32], u64>,
}

impl DedupCache {
    /// Create an empty deduplication cache.
    pub fn new() -> Self {
        Self {
            lookup: HashMap::new(),
        }
    }

    /// Return an existing canonical offset for this payload if present.
    pub fn get_canonical_offset(&self, payload: &[u8]) -> Option<u64> {
        let hash = blake3::hash(payload);
        self.lookup.get(hash.as_bytes()).copied()
    }

    /// Register a payload under the current offset and return whether it collides.
    pub fn register_payload(&mut self, payload: &[u8], offset: u64) -> Option<u64> {
        let hash = blake3::hash(payload);
        let bytes = *hash.as_bytes();
        if let Some(&existing) = self.lookup.get(&bytes) {
            Some(existing)
        } else {
            self.lookup.insert(bytes, offset);
            None
        }
    }
}

impl Default for DedupCache {
    fn default() -> Self {
        Self::new()
    }
}
