use crate::quantize::QuantLevel;
use crate::schema::TensorSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Single entry in the index describing a chunk's location.
pub struct IndexEntry {
    /// Chunk identifier.
    pub chunk_id: u64,
    /// Byte offset of the chunk header within the file.
    pub byte_offset: u64,
    /// Total byte length of the chunk (header + payload + checksum).
    pub byte_len: u64,
    /// Hash of the tensor name for quick lookup.
    pub tensor_name_hash: u64,
}

#[derive(Debug, Clone, Serialize)]
/// In-memory representation of the NCF index block.
pub struct NcfIndex {
    /// Number of entries in the index.
    pub entry_count: u64,
    /// Index entries for stored chunks.
    pub entries: Vec<IndexEntry>,
    /// Mapping from tensor name to primary chunk id.
    pub tensor_map: BTreeMap<String, u64>,
    /// Internal chunk ID lookup table for O(1) entry access.
    #[serde(skip)]
    pub chunk_map: HashMap<u64, usize>,
    /// Quantization levels assigned per tensor name.
    #[serde(default)]
    pub quant_levels: BTreeMap<String, QuantLevel>,
}

impl<'de> Deserialize<'de> for NcfIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NcfIndexHelper {
            entry_count: u64,
            entries: Vec<IndexEntry>,
            tensor_map: BTreeMap<String, u64>,
            #[serde(default)]
            quant_levels: BTreeMap<String, QuantLevel>,
        }

        let helper = NcfIndexHelper::deserialize(deserializer)?;
        let chunk_map = NcfIndex::build_chunk_map(&helper.entries);

        Ok(NcfIndex {
            entry_count: helper.entry_count,
            entries: helper.entries,
            tensor_map: helper.tensor_map,
            chunk_map,
            quant_levels: helper.quant_levels,
        })
    }
}

impl NcfIndex {
    fn build_chunk_map(entries: &[IndexEntry]) -> HashMap<u64, usize> {
        entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.chunk_id, idx))
            .collect()
    }

    /// Construct an index from entries and a tensor map.
    pub fn new(entries: Vec<IndexEntry>, tensor_map: BTreeMap<String, u64>) -> Self {
        let entry_count = entries.len() as u64;
        let chunk_map = Self::build_chunk_map(&entries);
        Self {
            entry_count,
            entries,
            tensor_map,
            chunk_map,
            quant_levels: BTreeMap::new(),
        }
    }

    /// Construct an index with explicit quantization metadata.
    pub fn with_quant_levels(
        entries: Vec<IndexEntry>,
        tensor_map: BTreeMap<String, u64>,
        quant_levels: BTreeMap<String, QuantLevel>,
    ) -> Self {
        let entry_count = entries.len() as u64;
        let chunk_map = Self::build_chunk_map(&entries);
        Self {
            entry_count,
            entries,
            tensor_map,
            chunk_map,
            quant_levels,
        }
    }

    /// Find the primary chunk id for a given tensor name.
    pub fn find_chunk_id(&self, name: &str) -> Option<u64> {
        self.tensor_map.get(name).copied()
    }

    /// Look up an index entry by chunk id in O(1).
    pub fn find_entry(&self, chunk_id: u64) -> Option<&IndexEntry> {
        self.chunk_map
            .get(&chunk_id)
            .and_then(|idx| self.entries.get(*idx))
    }

    /// Build a minimal index from a list of tensor schemas (first chunk per tensor).
    pub fn build_from_schemas(schemas: &[TensorSchema]) -> Self {
        let mut entries = Vec::new();
        let mut tensor_map = BTreeMap::new();
        for schema in schemas {
            if let Some(chunk_ref) = schema.chunks.first() {
                let name_hash = xxhash_rust::xxh3::xxh3_64(schema.name.as_bytes());
                entries.push(IndexEntry {
                    chunk_id: chunk_ref.chunk_id,
                    byte_offset: chunk_ref.byte_offset,
                    byte_len: chunk_ref.byte_len,
                    tensor_name_hash: name_hash,
                });
                tensor_map.insert(schema.name.clone(), chunk_ref.chunk_id);
            }
        }
        NcfIndex::new(entries, tensor_map)
    }
}
