use crate::prefetch::PrefetchReader;
use libc::{c_void, madvise, MADV_WILLNEED};
use memmap2::Mmap;
use ncf_core::constants::*;
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::IndexEntry;
use ncf_core::schema::TensorSchema;
use ncf_core::Result;
use once_cell::sync::Lazy;
use once_cell::sync::OnceCell;
use self_cell::self_cell;
use serde::Deserialize;
use serde_cbor::de::Deserializer as CborDeserializer;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Debug)]
/// Borrowed view of an NCF index decoded from the file.
pub struct BorrowedNcfIndex<'a> {
    /// Number of entries in the index.
    pub entry_count: u64,
    /// Index entries.
    pub entries: Vec<IndexEntry>,
    /// Mapping from tensor name to chunk id (borrowed str keys).
    pub tensor_map: HashMap<&'a str, u64>,
}

#[derive(Debug, Clone, Copy)]
/// Options controlling how an NCF reader is opened.
pub struct ReaderOptions {
    /// Enable prefetching on the reader.
    pub prefetch: bool,
    /// Verify all chunk checksums once when opening the reader.
    pub verify_on_open: bool,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            prefetch: false,
            verify_on_open: false,
        }
    }
}

/// A reader handle that can either use a direct NCF reader or a prefetch-aware reader.
pub enum NcfReaderHandle {
    /// A direct zero-copy reader.
    Direct(NcfReader),
    /// A reader with background prefetch support.
    Prefetch(PrefetchReader),
}

impl NcfReaderHandle {
    /// Open an NCF file with reader options.
    pub fn open_with_options<P: AsRef<Path>>(path: P, options: ReaderOptions) -> Result<Self> {
        if options.prefetch {
            let pre = PrefetchReader::open(path.as_ref())?;
            if options.verify_on_open {
                let _ = pre.verify_all_checksums()?;
            }
            Ok(NcfReaderHandle::Prefetch(pre))
        } else {
            let direct = NcfReader::open_with_options(path.as_ref(), options)?;
            Ok(NcfReaderHandle::Direct(direct))
        }
    }

    /// Return the decoded file metadata.
    pub fn metadata(&self) -> &NcfHeader {
        match self {
            NcfReaderHandle::Direct(reader) => reader.metadata(),
            NcfReaderHandle::Prefetch(reader) => reader.metadata(),
        }
    }

    /// Return the tensor schema list.
    pub fn schemas(&self) -> Result<&[TensorSchema]> {
        match self {
            NcfReaderHandle::Direct(reader) => reader.schemas(),
            NcfReaderHandle::Prefetch(reader) => reader.schemas().map_err(|err| err),
        }
    }

    /// Return the parsed NCF header prefix.
    pub fn header_prefix(&self) -> FileHeaderPrefix {
        match self {
            NcfReaderHandle::Direct(reader) => reader.header_prefix(),
            NcfReaderHandle::Prefetch(reader) => reader.header_prefix(),
        }
    }

    /// Return a zero-copy slice of the named tensor payload, if available.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        match self {
            NcfReaderHandle::Direct(reader) => reader.tensor_slice(name),
            NcfReaderHandle::Prefetch(reader) => reader.tensor_slice(name),
        }
    }

    /// Read the full tensor payload bytes.
    pub fn read_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        match self {
            NcfReaderHandle::Direct(reader) => reader.read_tensor(name),
            NcfReaderHandle::Prefetch(reader) => reader.read_tensor(name),
        }
    }

    /// Inspect the loaded NCF file and print metadata information.
    pub fn inspect(&self) -> Result<()> {
        match self {
            NcfReaderHandle::Direct(reader) => reader.inspect(),
            NcfReaderHandle::Prefetch(reader) => reader.inspect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawBorrowedNcfIndex<'a> {
    pub entry_count: u64,
    pub entries: Vec<IndexEntry>,
    #[serde(borrow)]
    tensor_map: BTreeMap<&'a str, u64>,
}

self_cell! {
    /// Reader that owns a memory map and exposes borrowed dependent data.
    pub struct NcfReader {
        owner: Mmap,
        #[covariant]
        dependent: NcfReaderData,
    }
}

#[derive(Debug)]
/// Owned dependent data stored alongside the memory map.
pub struct NcfReaderData<'this> {
    /// Parsed header metadata.
    pub metadata: NcfHeader,
    /// Lazily-initialized schema list.
    pub schemas: OnceCell<std::result::Result<Vec<TensorSchema>, String>>,
    /// Byte range of the schema block within the file.
    pub schema_range: std::ops::Range<usize>,
    /// Borrowed index data referencing the mapped memory.
    pub index: BorrowedNcfIndex<'this>,
    /// Parsed file header prefix.
    pub header_prefix: FileHeaderPrefix,
    /// Optional path key into the global parsed-header cache.
    pub cache_key: Option<PathBuf>,
    /// Auxiliary map from chunk_id -> index in `index.entries` for O(1) lookup.
    pub chunk_map: HashMap<u64, usize>,
}

struct CachedHeader {
    header_prefix: FileHeaderPrefix,
    metadata: NcfHeader,
    file_size: u64,
    schemas: OnceCell<std::result::Result<Vec<TensorSchema>, String>>,
}

static PARSED_HEADER_CACHE: Lazy<Mutex<HashMap<PathBuf, CachedHeader>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

impl NcfReader {
    /// Open an NCF file and return a reader providing borrowed access.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

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

        let reader = Self::try_new(mmap, |mmap| {
            // Attempt to use cached parsed header if available and file unchanged
            let key_path = path.as_ref().to_path_buf();
            let file_size = file.metadata()?.len();

            let mut use_cached = None;
            if let Ok(cache_guard) = PARSED_HEADER_CACHE.lock() {
                if let Some(ch) = cache_guard.get(&key_path) {
                    if ch.file_size == file_size {
                        use_cached = Some((ch.header_prefix.clone(), ch.metadata.clone()));
                    }
                }
            }

            // Gather any cached schemas now (used below)
            let mut cached_schemas: Option<std::result::Result<Vec<TensorSchema>, String>> = None;
            if let Ok(cache_guard) = PARSED_HEADER_CACHE.lock() {
                if let Some(ch) = cache_guard.get(&key_path) {
                    if ch.file_size == file_size {
                        if let Some(s) = ch.schemas.get() {
                            cached_schemas = Some(s.clone());
                        }
                    }
                }
            }

            let (header_prefix, metadata) = if let Some((hp, md)) = use_cached {
                (hp, md)
            } else {
                let header_prefix =
                    FileHeaderPrefix::decode(&mmap[..FILE_HEADER_PREFIX_SIZE as usize])
                        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
                let header_start = FILE_HEADER_PREFIX_SIZE as usize;
                let header_end = header_start
                    .checked_add(header_prefix.header_len as usize)
                    .ok_or_else(|| {
                        std::io::Error::new(ErrorKind::InvalidData, "header size overflow")
                    })?;
                if header_end > mmap.len() {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "header block out of bounds: end={}, file_size={}",
                            header_end,
                            mmap.len()
                        ),
                    ));
                }
                let metadata = NcfHeader::decode_cbor(&mmap[header_start..header_end])
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

                // store in cache (best-effort) with empty OnceCell for schemas
                if let Ok(mut cache_guard) = PARSED_HEADER_CACHE.lock() {
                    cache_guard.insert(
                        key_path.clone(),
                        CachedHeader {
                            header_prefix: header_prefix.clone(),
                            metadata: metadata.clone(),
                            file_size,
                            schemas: OnceCell::new(),
                        },
                    );
                }

                (header_prefix, metadata)
            };

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
                ));
            }
            let schema_range = schema_start..schema_end;
            // Prepare schemas OnceCell and populate from cache if available.
            let schemas_cell: OnceCell<std::result::Result<Vec<TensorSchema>, String>> =
                OnceCell::new();
            if let Some(sch) = cached_schemas {
                let _ = schemas_cell.set(sch);
            }

            const FOOTER_SIZE: usize = 16; // 8 bytes magic + 8 bytes length
            if mmap.len() < FOOTER_SIZE {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "file too small to contain footer",
                ));
            }

            let footer_position = mmap.len() - FOOTER_SIZE;
            let footer_magic = &mmap[footer_position..footer_position + 8];
            if footer_magic != b"NCFEND!!" {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing or invalid footer magic",
                ));
            }

            let footer_len_bytes: [u8; 8] = mmap[footer_position + 8..footer_position + 16]
                .try_into()
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid footer length")
                })?;
            let index_len = u64::from_le_bytes(footer_len_bytes) as usize;
            let index_start = header_prefix.index_offset as usize;
            let index_end = index_start.checked_add(index_len).ok_or_else(|| {
                std::io::Error::new(ErrorKind::InvalidData, "index size overflow")
            })?;

            if index_end > footer_position {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "index block overlaps footer: end={}, footer_pos={}",
                        index_end, footer_position
                    ),
                ));
            }

            let mut index_de = CborDeserializer::from_slice(&mmap[index_start..index_end]);
            let raw_index: RawBorrowedNcfIndex<'_> = Deserialize::deserialize(&mut index_de)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
            let mut tensor_map = HashMap::with_capacity(raw_index.tensor_map.len());
            for (name, chunk_id) in &raw_index.tensor_map {
                tensor_map.insert(*name, *chunk_id);
            }

            let index = BorrowedNcfIndex {
                entry_count: raw_index.entry_count,
                entries: raw_index.entries,
                tensor_map,
            };

            // build chunk_map for fast lookup
            let mut chunk_map = HashMap::with_capacity(index.entries.len());
            for (i, e) in index.entries.iter().enumerate() {
                chunk_map.insert(e.chunk_id, i);
            }

            Ok(NcfReaderData {
                metadata,
                schemas: schemas_cell,
                schema_range,
                index,
                header_prefix,
                cache_key: Some(key_path.clone()),
                chunk_map,
            })
        })?;

        Ok(reader)
    }

    /// Open with `ReaderOptions` control (e.g., verify_on_open).
    pub fn open_with_options<P: AsRef<Path>>(path: P, options: ReaderOptions) -> Result<Self> {
        let reader = NcfReader::open(path.as_ref())?;
        if options.verify_on_open {
            let _ = reader.verify_all_checksums()?;
        }
        Ok(reader)
    }

    /// Print basic info about the NCF file to stdout (for debugging).
    pub fn inspect(&self) -> Result<()> {
        let schemas = self.schemas()?;
        println!(
            "Model: {}",
            self.borrow_dependent().metadata.metadata.model_name
        );
        println!(
            "Architecture: {}",
            self.borrow_dependent().metadata.metadata.architecture
        );
        println!("Tensors: {}", schemas.len());
        for tensor in schemas.iter() {
            println!(" - {} {} {:?}", tensor.name, tensor.dtype, tensor.shape);
        }
        Ok(())
    }

    /// Find a tensor schema by name.
    pub fn find_schema(&self, name: &str) -> Result<Option<&TensorSchema>> {
        Ok(self.schemas()?.iter().find(|schema| schema.name == name))
    }

    /// Return the decoded NCF header metadata.
    pub fn metadata(&self) -> &NcfHeader {
        &self.borrow_dependent().metadata
    }

    /// Return the number of schemas/tensors in the file.
    pub fn schema_count(&self) -> Result<usize> {
        Ok(self.borrow_dependent().index.tensor_map.len())
    }

    /// Return a zero-copy slice for a tensor payload by name, if present.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_id = self.borrow_dependent().index.tensor_map.get(name)?;
        let idx = *self.borrow_dependent().chunk_map.get(chunk_id)?;
        let entry = &self.borrow_dependent().index.entries[idx];
        let data = self.borrow_owner();

        let offset_start = (entry.byte_offset as usize).checked_add(CHUNK_HEADER_SIZE as usize)?;
        if offset_start > data.len() {
            return None;
        }

        let chunk_total_len = entry.byte_len as usize;
        let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;
        if chunk_total_len < chunk_overhead {
            return None;
        }

        let data_len = chunk_total_len - chunk_overhead;
        let offset_end = offset_start.checked_add(data_len)?;
        if offset_end > data.len() {
            return None;
        }

        // Advise kernel to prefetch region to reduce page faults on large reads
        if offset_end > offset_start {
            unsafe {
                let len = offset_end - offset_start;
                let ptr = data.as_ptr().add(offset_start) as *mut c_void;
                let _ = madvise(ptr, len, MADV_WILLNEED);
            }
        }

        Some(&data[offset_start..offset_end])
    }

    /// Return the parsed file header prefix.
    pub fn header_prefix(&self) -> FileHeaderPrefix {
        self.borrow_dependent().header_prefix
    }

    /// Lazily decode and return the tensor schemas.
    pub fn schemas(&self) -> Result<&[TensorSchema]> {
        self.with_dependent(|owner, data| {
            let schemas_cell = data.schemas.get_or_init(|| {
                let schema_bytes = &owner[data.schema_range.clone()];
                let mut schema_de = CborDeserializer::from_slice(schema_bytes);
                Deserialize::deserialize(&mut schema_de).map_err(|err| err.to_string())
            });

            // If we have a global cache entry for this path, populate it once
            if let Some(cache_key) = &data.cache_key {
                if let Ok(cache_guard) = PARSED_HEADER_CACHE.lock() {
                    if let Some(ch) = cache_guard.get(cache_key) {
                        // best-effort: set global OnceCell if not already set
                        if ch.schemas.get().is_none() {
                            let _ = ch.schemas.set(schemas_cell.clone());
                        }
                    }
                }
            }

            match schemas_cell.as_ref() {
                Ok(schemas) => Ok(schemas.as_slice()),
                Err(err) => Err(ncf_core::NcfError::Header(err.clone())),
            }
        })
    }

    /// Read and return the full tensor payload bytes for the given name.
    pub fn read_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let schema = match self.find_schema(name)? {
            Some(schema) => schema,
            None => return Ok(None),
        };

        let data = self.borrow_owner();
        let mut result = Vec::new();

        for chunk in &schema.chunks {
            // Bounds check: chunk offset is within file
            let offset_start = (chunk.byte_offset as usize)
                .checked_add(CHUNK_HEADER_SIZE as usize)
                .ok_or_else(|| {
                    std::io::Error::new(ErrorKind::InvalidData, "chunk offset overflow")
                })?;

            if offset_start > data.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "chunk offset out of bounds: offset={}, file_size={}",
                        offset_start,
                        data.len()
                    ),
                )
                .into());
            }

            // Calculate actual data length: total_len - header - checksum
            let chunk_total_len = chunk.byte_len as usize;
            let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;

            if chunk_total_len < chunk_overhead {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "chunk size too small: total_len={}, overhead={}",
                        chunk_total_len, chunk_overhead
                    ),
                )
                .into());
            }

            let data_len = chunk_total_len - chunk_overhead;

            // Bounds check: slice end is within file
            let offset_end = offset_start.checked_add(data_len).ok_or_else(|| {
                std::io::Error::new(ErrorKind::InvalidData, "chunk data size overflow")
            })?;

            if offset_end > data.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "chunk data out of bounds: end={}, file_size={}",
                        offset_end,
                        data.len()
                    ),
                )
                .into());
            }

            result.extend_from_slice(&data[offset_start..offset_end]);
        }
        Ok(Some(result))
    }

    /// Read quantized tensor values based on the tensor dtype's quantization format.
    /// This uses a pure enum match on DType and avoids trait objects or vtable dispatch.
    pub fn read_tensor_quantized_values(&self, name: &str) -> Result<Option<Vec<u32>>> {
        let schema = match self.find_schema(name)? {
            Some(schema) => schema,
            None => return Ok(None),
        };

        if !schema.dtype.is_quantized() {
            return Ok(None);
        }

        let slice = match self.tensor_slice(name) {
            Some(bytes) => bytes,
            None => return Ok(None),
        };

        let element_count = schema.shape.iter().copied().product::<u64>() as usize;
        let quantized =
            ncf_core::quantize::unpack_quantized_payload(schema.dtype, slice, element_count)
                .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err))?;
        Ok(Some(quantized))
    }

    /// Verify all chunk payload checksums once. This computes Blake3 over each
    /// payload and compares against the checksum recorded in the schema chunk
    /// references. This is intentionally an explicit method (not automatic).
    pub fn verify_all_checksums(&self) -> Result<()> {
        let data = self.borrow_owner();
        let schemas = self.schemas()?;
        for s in schemas.iter() {
            for c in s.chunks.iter() {
                let idx = match self.borrow_dependent().chunk_map.get(&c.chunk_id) {
                    Some(i) => *i,
                    None => continue,
                };
                let entry = &self.borrow_dependent().index.entries[idx];
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
                if offset_end > data.len() {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        "chunk data out of bounds",
                    )
                    .into());
                }
                let payload = &data[offset_start..offset_end];
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
