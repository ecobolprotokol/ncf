use bytes::Bytes;
use libc::{c_void, madvise, MADV_WILLNEED};
use memmap2::Mmap;
use ncf_core::constants::{CHUNK_CHECKSUM_SIZE, CHUNK_HEADER_SIZE, FILE_HEADER_PREFIX_SIZE};
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::NcfIndex;
use ncf_core::schema::TensorSchema;
use ncf_core::Result;
use serde::Deserialize;
use serde_cbor::de::Deserializer as CborDeserializer;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

/// Options controlling how the prefetch reader opens an NCF file.
pub struct PrefetchOptions {
    /// Whether to perform background prefetching.
    pub enabled: bool,
}

impl Default for PrefetchOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Reader wrapper that prefetches the next tensor chunk into a shared buffer.
pub struct PrefetchReader {
    mmap: Arc<Mmap>,
    header_prefix: FileHeaderPrefix,
    metadata: NcfHeader,
    schemas: Vec<TensorSchema>,
    index: Arc<NcfIndex>,
    buffer: Arc<Mutex<HashMap<u64, Bytes>>>,
}

impl PrefetchReader {
    /// Open an NCF file and prepare the prefetch buffer.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        if (mmap.len() as u64) < FILE_HEADER_PREFIX_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!("file too small: {} bytes, need at least {}", mmap.len(), FILE_HEADER_PREFIX_SIZE)
            ).into());
        }

        let header_prefix = FileHeaderPrefix::decode(&mmap[..FILE_HEADER_PREFIX_SIZE as usize])
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err.to_string()))?;

        let header_start = FILE_HEADER_PREFIX_SIZE as usize;
        let header_end = header_start.checked_add(header_prefix.header_len as usize)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "header size overflow"))?;
        let metadata = NcfHeader::decode_cbor(&mmap[header_start..header_end])?;

        let schema_start = header_prefix.schema_offset as usize;
        let schema_end = header_prefix.index_offset as usize;
        let schema_bytes = &mmap[schema_start..schema_end];
        let mut schema_de = CborDeserializer::from_slice(schema_bytes);
        let schemas: Vec<TensorSchema> = Deserialize::deserialize(&mut schema_de)
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err))?;

        let footer_position = mmap.len() - 16;
        let index_len = u64::from_le_bytes(mmap[footer_position + 8..footer_position + 16].try_into().unwrap()) as usize;
        let index_start = header_prefix.index_offset as usize;
        let index_end = index_start.checked_add(index_len)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "index size overflow"))?;

        let mut index_de = CborDeserializer::from_slice(&mmap[index_start..index_end]);
        let index: NcfIndex = Deserialize::deserialize(&mut index_de)
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err))?;

        Ok(Self {
            mmap,
            header_prefix,
            metadata,
            schemas,
            index: Arc::new(index),
            buffer: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn advise_region(&self, start: usize, len: usize) {
        if len == 0 || start >= self.mmap.len() {
            return;
        }
        let len = len.min(self.mmap.len() - start);
        unsafe {
            let ptr = self.mmap.as_ptr().add(start) as *mut c_void;
            let _ = madvise(ptr, len, MADV_WILLNEED);
        }
    }

    fn spawn_prefetch_for(&self, current_offset: u64) {
        let next_entry = self.index.entries.iter()
            .filter(|entry| entry.byte_offset > current_offset)
            .min_by_key(|entry| entry.byte_offset)
            .cloned();

        if let Some(entry) = next_entry {
            let mmap = Arc::clone(&self.mmap);
            let buffer = Arc::clone(&self.buffer);
            thread::spawn(move || {
                let offset_start = entry.byte_offset as usize + CHUNK_HEADER_SIZE as usize;
                if offset_start >= mmap.len() {
                    return;
                }
                let data_len = (entry.byte_len as usize).saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
                let offset_end = offset_start.saturating_add(data_len);
                if offset_end > mmap.len() {
                    return;
                }
                let payload = Bytes::copy_from_slice(&mmap[offset_start..offset_end]);
                let mut guard = buffer.lock().unwrap();
                guard.insert(entry.chunk_id, payload);
            });
        }
    }

    /// Read a tensor payload and start background prefetch for the next tensor chunk.
    pub fn metadata(&self) -> &NcfHeader {
        &self.metadata
    }

    /// Return the tensor schema list from the opened NCF file.
    pub fn schemas(&self) -> Result<&[TensorSchema]> {
        Ok(&self.schemas)
    }

    /// Return the parsed NCF header prefix for the opened file.
    pub fn header_prefix(&self) -> FileHeaderPrefix {
        self.header_prefix
    }

    /// Return a zero-copy slice of a named tensor payload.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_id = self.index.find_chunk_id(name)?;
        let entry = self.index.entries.iter().find(|entry| entry.chunk_id == chunk_id)?;
        let offset_start = entry.byte_offset as usize + CHUNK_HEADER_SIZE as usize;
        let data_len = (entry.byte_len as usize).saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
        let offset_end = offset_start.checked_add(data_len)?;
        if offset_end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[offset_start..offset_end])
    }

    /// Print file metadata and schema details for debugging.
    pub fn inspect(&self) -> Result<()> {
        let schemas = self.schemas()?;
        println!("Model: {}", self.metadata.metadata.model_name);
        println!("Architecture: {}", self.metadata.metadata.architecture);
        println!("Tensors: {}", schemas.len());
        for tensor in schemas.iter() {
            println!(" - {} {} {:?}", tensor.name, tensor.dtype, tensor.shape);
        }
        Ok(())
    }

    /// Read a tensor payload and start background prefetch for the next chunk.
    pub fn read_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let chunk_id = self.index.find_chunk_id(name);
        let chunk_id = match chunk_id {
            Some(id) => id,
            None => return Ok(None),
        };

        let entry = self.index.entries.iter().find(|entry| entry.chunk_id == chunk_id);
        let entry = match entry {
            Some(entry) => entry,
            None => return Ok(None),
        };

        self.spawn_prefetch_for(entry.byte_offset);
        self.advise_region(entry.byte_offset as usize, entry.byte_len as usize);

        if let Some(bytes) = self.buffer.lock().unwrap().remove(&entry.chunk_id) {
            return Ok(Some(bytes.to_vec()));
        }

        let data = &self.mmap;
        let offset_start = entry.byte_offset as usize + CHUNK_HEADER_SIZE as usize;
        let data_len = (entry.byte_len as usize).saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
        let offset_end = offset_start + data_len;
        if offset_end > data.len() {
            return Err(std::io::Error::new(ErrorKind::InvalidData, "chunk data out of bounds").into());
        }

        Ok(Some(data[offset_start..offset_end].to_vec()))
    }
}
