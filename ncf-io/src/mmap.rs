use libc::{c_void, madvise, MADV_WILLNEED};
use memmap2::Mmap;
use ncf_core::constants::*;
use ncf_core::header::FileHeaderPrefix;
use ncf_core::index::NcfIndex;
use ncf_core::schema::TensorSchema;
use ncf_core::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, ErrorKind};
use std::path::Path;
use std::sync::OnceLock;

/// Memory-mapped view of an NCF file for zero-copy reads.
pub struct NcfMmap {
    /// Underlying memory map of the file.
    pub mmap: Mmap,
    /// Parsed file header prefix.
    pub header_prefix: FileHeaderPrefix,
    /// Decoded CBOR header metadata.
    pub metadata: ncf_core::header::NcfHeader,
    schemas: OnceLock<std::result::Result<Vec<TensorSchema>, String>>,
    schema_range: std::ops::Range<usize>,
    /// Parsed index information.
    pub index: NcfIndex,
    /// Auxiliary map from chunk_id -> index in `entries` for O(1) lookup.
    chunk_map: HashMap<u64, usize>,
}

impl NcfMmap {
    /// Open and memory-map the given file path as an NCF file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Bounds check: minimum file size
        if (mmap.len() as u64) < FILE_HEADER_PREFIX_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "file too small: {} bytes, need at least {}",
                    mmap.len(),
                    FILE_HEADER_PREFIX_SIZE
                ),
            )
            .into());
        }

        // Decode header prefix (first 48 bytes)
        let header_prefix = FileHeaderPrefix::decode(&mmap[..FILE_HEADER_PREFIX_SIZE as usize])?;

        // Bounds check: header block
        let header_start = FILE_HEADER_PREFIX_SIZE as usize;
        let header_end = header_start
            .checked_add(header_prefix.header_len as usize)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "header size overflow"))?;
        if header_end > mmap.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "header block out of bounds: end={}, file_size={}",
                    header_end,
                    mmap.len()
                ),
            )
            .into());
        }

        let metadata = ncf_core::header::NcfHeader::decode_cbor(&mmap[header_start..header_end])?;

        // Bounds check: schema block
        let schema_start = header_prefix.schema_offset as usize;
        let schema_end = header_prefix.index_offset as usize;
        if schema_start > schema_end || schema_end > mmap.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "schema block out of bounds: start={}, end={}, file_size={}",
                    schema_start,
                    schema_end,
                    mmap.len()
                ),
            )
            .into());
        }
        let schema_range = schema_start..schema_end;

        // Bounds check: footer
        const FOOTER_SIZE: usize = 16; // 8 bytes magic + 8 bytes length
        if mmap.len() < FOOTER_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "file too small to contain footer",
            )
            .into());
        }

        let footer_position = mmap.len() - FOOTER_SIZE;
        let footer_magic = &mmap[footer_position..footer_position + 8];
        if footer_magic != b"NCFEND!!" {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "missing or invalid footer magic",
            )
            .into());
        }

        let footer_len_bytes: [u8; 8] = mmap[footer_position + 8..footer_position + 16]
            .try_into()
            .map_err(|_| std::io::Error::new(ErrorKind::InvalidData, "invalid footer length"))?;
        let index_len = u64::from_le_bytes(footer_len_bytes) as usize;

        // Bounds check: index block
        let index_start = header_prefix.index_offset as usize;
        let index_end = index_start
            .checked_add(index_len)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "index size overflow"))?;
        if index_end > footer_position {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "index block overlaps footer: end={}, footer_pos={}",
                    index_end, footer_position
                ),
            )
            .into());
        }

        let index: NcfIndex =
            ciborium::de::from_reader(Cursor::new(&mmap[index_start..index_end]))?;

        // build chunk_map for fast chunk id -> entry lookup
        let mut chunk_map = HashMap::with_capacity(index.entries.len());
        for (i, e) in index.entries.iter().enumerate() {
            chunk_map.insert(e.chunk_id, i);
        }

        Ok(Self {
            mmap,
            header_prefix,
            metadata,
            schemas: OnceLock::new(),
            schema_range,
            index,
            chunk_map,
        })
    }

    /// Lazily decode and return the list of tensor schemas.
    pub fn schemas(&self) -> Result<&[TensorSchema]> {
        let schemas = self.schemas.get_or_init(|| {
            let slice = &self.mmap[self.schema_range.clone()];
            ciborium::de::from_reader(Cursor::new(slice)).map_err(|err| err.to_string())
        });

        match schemas.as_ref() {
            Ok(schemas) => Ok(schemas.as_slice()),
            Err(err) => Err(ncf_core::NcfError::Header(err.clone())),
        }
    }

    /// Return a zero-copy slice of the tensor payload for the given name.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_id = self.index.tensor_map.get(name)?;
        let idx = *self.chunk_map.get(chunk_id)?;
        let entry = &self.index.entries[idx];

        // Bounds check: chunk offset is within file
        let offset_start = (entry.byte_offset as usize).checked_add(CHUNK_HEADER_SIZE as usize)?;
        if offset_start > self.mmap.len() {
            return None;
        }

        // Calculate actual data length: total_len - header - checksum
        let chunk_total_len = entry.byte_len as usize;
        let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;

        if chunk_total_len < chunk_overhead {
            return None;
        }

        let data_len = chunk_total_len - chunk_overhead;

        // Bounds check: slice end is within file
        let offset_end = offset_start.checked_add(data_len)?;
        if offset_end > self.mmap.len() {
            return None;
        }

        // Advise the kernel to prefetch the region to reduce page faults
        if offset_end > offset_start {
            unsafe {
                let len = offset_end - offset_start;
                let ptr = self.mmap.as_ptr().add(offset_start) as *mut c_void;
                let _ = madvise(ptr, len, MADV_WILLNEED);
            }
        }

        Some(&self.mmap[offset_start..offset_end])
    }

    /// Verify all chunk payload checksums once. This computes Blake3 over each
    /// payload and compares against the checksum recorded in the schema chunk
    /// references. This is intentionally an explicit method (not automatic).
    pub fn verify_all_checksums(&self) -> Result<()> {
        let schemas = self.schemas()?;
        for s in schemas.iter() {
            for c in s.chunks.iter() {
                let idx = match self.chunk_map.get(&c.chunk_id) {
                    Some(i) => *i,
                    None => continue,
                };
                let entry = &self.index.entries[idx];
                let offset_start = (entry.byte_offset as usize)
                    .checked_add(CHUNK_HEADER_SIZE as usize)
                    .ok_or_else(|| {
                        std::io::Error::new(ErrorKind::InvalidData, "chunk offset overflow")
                    })?;
                let data_len = (entry.byte_len as usize)
                    .saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
                let offset_end = offset_start.checked_add(data_len).ok_or_else(|| {
                    std::io::Error::new(ErrorKind::InvalidData, "chunk data size overflow")
                })?;
                if offset_end > self.mmap.len() {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        "chunk data out of bounds",
                    )
                    .into());
                }
                let payload = &self.mmap[offset_start..offset_end];
                let hash = blake3::hash(payload);
                if hash.as_bytes() != &c.checksum {
                    return Err(
                        std::io::Error::new(ErrorKind::InvalidData, "checksum mismatch").into(),
                    );
                }
            }
        }
        Ok(())
    }
}
