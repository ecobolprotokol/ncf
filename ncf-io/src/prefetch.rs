use bytes::Bytes;
use crossbeam::channel::{unbounded, Sender};
use libc::{c_void, posix_madvise, POSIX_MADV_WILLNEED};
use memmap2::Mmap;
use ncf_core::constants::{CHUNK_CHECKSUM_SIZE, CHUNK_HEADER_SIZE, FILE_HEADER_PREFIX_SIZE};
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::{IndexEntry, NcfIndex};
use ncf_core::schema::TensorSchema;
use ncf_core::Result;
use serde::Deserialize;
use serde_cbor::de::Deserializer as CborDeserializer;
use std::collections::{HashMap, HashSet};
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
    task_tx: Sender<PrefetchTask>,
    workers: Vec<thread::JoinHandle<()>>,
}

enum PrefetchTask {
    PredictNext(u64),
}

impl PrefetchReader {
    const PREFETCH_WORKER_COUNT: usize = 2;

    /// Open an NCF file and prepare the prefetch buffer.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

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

        let header_prefix = FileHeaderPrefix::decode(&mmap[..FILE_HEADER_PREFIX_SIZE as usize])
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err.to_string()))?;

        let header_start = FILE_HEADER_PREFIX_SIZE as usize;
        let header_end = header_start
            .checked_add(header_prefix.header_len as usize)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "header size overflow"))?;
        let metadata = NcfHeader::decode_cbor(&mmap[header_start..header_end])?;

        let schema_start = header_prefix.schema_offset as usize;
        let schema_end = header_prefix.index_offset as usize;
        let schema_bytes = &mmap[schema_start..schema_end];
        let mut schema_de = CborDeserializer::from_slice(schema_bytes);
        let schemas: Vec<TensorSchema> = Deserialize::deserialize(&mut schema_de)
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err))?;

        let footer_position = mmap.len() - 16;
        let index_len = u64::from_le_bytes(
            mmap[footer_position + 8..footer_position + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let index_start = header_prefix.index_offset as usize;
        let index_end = index_start
            .checked_add(index_len)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "index size overflow"))?;

        let mut index_de = CborDeserializer::from_slice(&mmap[index_start..index_end]);
        let index: NcfIndex = Deserialize::deserialize(&mut index_de)
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err))?;
        let index = Arc::new(index);

        let buffer = Arc::new(Mutex::new(HashMap::new()));
        let inflight = Arc::new(Mutex::new(HashSet::new()));
        let (task_tx, task_rx) = unbounded::<PrefetchTask>();

        let mut workers = Vec::with_capacity(Self::PREFETCH_WORKER_COUNT);
        for worker_id in 0..Self::PREFETCH_WORKER_COUNT {
            let mmap_clone = Arc::clone(&mmap);
            let index_clone = Arc::clone(&index);
            let buffer_clone = Arc::clone(&buffer);
            let inflight_clone = Arc::clone(&inflight);
            let task_rx_clone = task_rx.clone();

            let handle = thread::Builder::new()
                .name(format!("ncf-prefetch-worker-{}", worker_id))
                .spawn(move || {
                    for task in task_rx_clone.iter() {
                        let PrefetchTask::PredictNext(current_offset) = task;
                        let candidate = index_clone
                            .entries
                            .iter()
                            .filter(|entry| entry.byte_offset > current_offset)
                            .min_by_key(|entry| entry.byte_offset)
                            .cloned();

                        if let Some(entry) = candidate {
                            Self::prefetch_entry(
                                &mmap_clone,
                                &buffer_clone,
                                &inflight_clone,
                                &entry,
                            );
                        }
                    }
                })?;

            workers.push(handle);
        }

        Ok(Self {
            mmap,
            header_prefix,
            metadata,
            schemas,
            index,
            buffer,
            task_tx,
            workers,
        })
    }

    /// Verify all chunk payload checksums once for the opened file.
    pub fn verify_all_checksums(&self) -> Result<()> {
        let data = &*self.mmap;
        for s in self.schemas.iter() {
            for c in s.chunks.iter() {
                // find entry by chunk id
                let entry_opt = self.index.entries.iter().find(|e| e.chunk_id == c.chunk_id);
                let entry = match entry_opt {
                    Some(e) => e,
                    None => continue,
                };
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

    fn advise_region(&self, start: usize, len: usize) {
        Self::advise_region_static(&self.mmap, start, len)
    }

    fn advise_region_static(mmap: &Mmap, start: usize, len: usize) {
        if len == 0 || start >= mmap.len() {
            return;
        }

        let len = len.min(mmap.len() - start);
        unsafe {
            let ptr = mmap.as_ptr().add(start) as *mut c_void;
            let _ = posix_madvise(ptr, len, POSIX_MADV_WILLNEED);
        }
    }

    fn prefetch_entry(
        mmap: &Mmap,
        buffer: &Arc<Mutex<HashMap<u64, Bytes>>>,
        inflight: &Arc<Mutex<HashSet<u64>>>,
        entry: &IndexEntry,
    ) {
        {
            let mut guard = inflight.lock().unwrap();
            if !guard.insert(entry.chunk_id) {
                return;
            }
        }

        Self::advise_region_static(mmap, entry.byte_offset as usize, entry.byte_len as usize);
        let offset_start = entry.byte_offset as usize + CHUNK_HEADER_SIZE as usize;
        if offset_start >= mmap.len() {
            let mut guard = inflight.lock().unwrap();
            guard.remove(&entry.chunk_id);
            return;
        }

        let data_len = (entry.byte_len as usize)
            .saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
        let offset_end = offset_start.saturating_add(data_len);
        if offset_end > mmap.len() {
            let mut guard = inflight.lock().unwrap();
            guard.remove(&entry.chunk_id);
            return;
        }

        let payload = Bytes::copy_from_slice(&mmap[offset_start..offset_end]);
        let mut buffer_guard = buffer.lock().unwrap();
        buffer_guard.entry(entry.chunk_id).or_insert(payload);

        let mut guard = inflight.lock().unwrap();
        guard.remove(&entry.chunk_id);
    }

    fn spawn_prefetch_for(&self, current_offset: u64) {
        let _ = self
            .task_tx
            .try_send(PrefetchTask::PredictNext(current_offset));
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
        let entry = self
            .index
            .entries
            .iter()
            .find(|entry| entry.chunk_id == chunk_id)?;
        let offset_start = entry.byte_offset as usize + CHUNK_HEADER_SIZE as usize;
        let data_len = (entry.byte_len as usize)
            .saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
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

        let entry = self
            .index
            .entries
            .iter()
            .find(|entry| entry.chunk_id == chunk_id);
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
        let data_len = (entry.byte_len as usize)
            .saturating_sub((CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize);
        let offset_end = offset_start + data_len;
        if offset_end > data.len() {
            return Err(
                std::io::Error::new(ErrorKind::InvalidData, "chunk data out of bounds").into(),
            );
        }

        Ok(Some(data[offset_start..offset_end].to_vec()))
    }
}

impl Drop for PrefetchReader {
    fn drop(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
