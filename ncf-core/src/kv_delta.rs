use serde::{Deserialize, Serialize};

/// Encoding mode for a KV cache chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KvEncoding {
    /// A full snapshot of the KV cache.
    Full,
    /// A delta record relative to a previous snapshot.
    Delta {
        /// Offset into the stream or snapshot base for delta reconstruction.
        base_offset: u64,
    },
}

/// A single encoded KV cache record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvChunk {
    /// Token index for this KV chunk.
    pub token_index: u64,
    /// Encoding metadata.
    pub kv_encoding: KvEncoding,
    /// Compressed or full payload bytes.
    pub payload: Vec<u8>,
}

impl KvChunk {
    /// Build a KV full snapshot record.
    pub fn full_snapshot(token_index: u64, payload: Vec<u8>) -> Self {
        Self {
            token_index,
            kv_encoding: KvEncoding::Full,
            payload,
        }
    }

    /// Build a delta-encoded chunk.
    pub fn delta_snapshot(token_index: u64, base_offset: u64, payload: Vec<u8>) -> Self {
        Self {
            token_index,
            kv_encoding: KvEncoding::Delta { base_offset },
            payload,
        }
    }
}

/// Encode KV cache chunks into delta-encoded records.
pub struct KvDeltaEncoder;

impl KvDeltaEncoder {
    /// Create a delta encoding chain from a sequence of KV payloads.
    pub fn encode(payloads: &[Vec<u8>]) -> std::result::Result<Vec<KvChunk>, std::io::Error> {
        let mut output = Vec::with_capacity(payloads.len());
        let mut last_full_index = 0u64;

        for (idx, chunk) in payloads.iter().enumerate() {
            let token_index = idx as u64;
            if idx % 1000 == 0 {
                output.push(KvChunk::full_snapshot(token_index, chunk.clone()));
                last_full_index = token_index;
                continue;
            }

            let previous_chunk = &payloads[idx - 1];
            let delta: Vec<u8> = previous_chunk
                .iter()
                .zip(chunk.iter())
                .map(|(a, b)| a ^ b)
                .collect();
            let compressed = zstd::encode_all(delta.as_slice(), 0)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
            output.push(KvChunk::delta_snapshot(
                token_index,
                last_full_index,
                compressed,
            ));
        }

        Ok(output)
    }

    /// Decode a delta chain back to a full payload for the requested token index.
    pub fn decode(
        chain: &[KvChunk],
        target_index: u64,
    ) -> std::result::Result<Vec<u8>, std::io::Error> {
        let mut reconstructed = None;
        for chunk in chain.iter() {
            if chunk.token_index > target_index {
                break;
            }
            match &chunk.kv_encoding {
                KvEncoding::Full => reconstructed = Some(chunk.payload.clone()),
                KvEncoding::Delta { base_offset: _ } => {
                    if let Some(previous) = reconstructed.take() {
                        let delta = zstd::decode_all(&chunk.payload[..])
                            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
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

        reconstructed.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("missing full KV snapshot for target index {}", target_index),
            )
        })
    }
}
