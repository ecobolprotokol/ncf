use ncf_core::Result;
use ncf_core::kv_delta::{KvChunk, KvEncoding};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const RAM_THRESHOLD_BYTES: usize = 14 * 1024 * 1024 * 1024;

/// A simple SSD-backed LRU cache for reconstructed KV payloads.
pub struct KvStreamReader {
    cache: HashMap<u64, Vec<u8>>,
    lru_order: VecDeque<u64>,
    disk_index: HashMap<u64, PathBuf>,
    temp_dir: PathBuf,
    current_usage: usize,
}

impl KvStreamReader {
    /// Create a new KV stream reader using the provided temporary directory.
    pub fn new<P: AsRef<Path>>(temp_dir: P) -> Self {
        KvStreamReader {
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            disk_index: HashMap::new(),
            temp_dir: temp_dir.as_ref().to_path_buf(),
            current_usage: 0,
        }
    }

    fn evict_if_needed(&mut self) -> Result<()> {
        while self.current_usage > RAM_THRESHOLD_BYTES {
            if let Some(oldest) = self.lru_order.pop_front() {
                if let Some(payload) = self.cache.remove(&oldest) {
                    let path = self.temp_dir.join(format!("ncf_kv_{}.cache", oldest));
                    let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(&path)?;
                    file.write_all(&payload)?;
                    self.disk_index.insert(oldest, path);
                    self.current_usage = self.current_usage.saturating_sub(payload.len());
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    fn touch(&mut self, token_index: u64) {
        self.lru_order.retain(|&id| id != token_index);
        self.lru_order.push_back(token_index);
    }

    fn load_from_disk(&mut self, token_index: u64) -> Result<Option<Vec<u8>>> {
        if let Some(path) = self.disk_index.remove(&token_index) {
            let mut file = File::open(&path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            let usage = buffer.len();
            self.cache.insert(token_index, buffer.clone());
            self.current_usage += usage;
            let _ = std::fs::remove_file(path);
            self.touch(token_index);
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }

    /// Reconstruct a full KV payload chain from snapshot and delta records.
    pub fn reconstruct(&mut self, chain: &[KvChunk], target_index: u64) -> Result<Vec<u8>> {
        if let Some(payload) = self.cache.get(&target_index).cloned() {
            self.touch(target_index);
            return Ok(payload);
        }

        if let Some(payload) = self.load_from_disk(target_index)? {
            return Ok(payload);
        }

        let mut reconstructed = None;
        for chunk in chain.iter() {
            if chunk.token_index > target_index {
                break;
            }
            match &chunk.kv_encoding {
                KvEncoding::Full => reconstructed = Some(chunk.payload.clone()),
                KvEncoding::Delta { .. } => {
                    if let Some(previous) = reconstructed.take() {
                        let delta = zstd::decode_all(&chunk.payload[..])?;
                        let next: Vec<u8> = previous
                            .iter()
                            .zip(delta.iter())
                            .map(|(a, b)| a ^ b)
                            .collect();
                        reconstructed = Some(next);
                    }
                }
            }
        }

        let payload = reconstructed.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing full KV snapshot"))?;
        let usage = payload.len();
        self.cache.insert(target_index, payload.clone());
        self.touch(target_index);
        self.current_usage += usage;
        self.evict_if_needed()?;
        Ok(payload)
    }
}
