use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Aligned vector with a 64-byte heap allocation suitable for AVX-512 access.
pub struct AlignedVec {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

impl AlignedVec {
    /// Allocate a 64-byte aligned copy of the given data.
    pub fn new(data: &[u8]) -> Self {
        let len = data.len();
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
            };
        }

        let layout = Layout::from_size_align(len, 64)
            .expect("failed to create aligned layout");
        let raw_ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw_ptr).expect("allocation failed");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), len);
        }

        Self { ptr, len, cap: len }
    }

    /// Return the aligned bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Return the aligned bytes as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = Layout::from_size_align(self.cap, 64)
                .expect("failed to create aligned layout");
            unsafe { dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}

impl std::fmt::Debug for AlignedVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("cap", &self.cap)
            .finish()
    }
}

/// A compact metadata structure for tensor chunks that records AVX-512 width.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkMetadata {
    /// Chunk identifier.
    pub chunk_id: u64,
    /// Byte offset of the chunk within the file.
    pub byte_offset: u64,
    /// Total chunk byte length.
    pub byte_len: u64,
    /// SIMD width in bytes used to align the chunk payload.
    pub simd_width: u32,
}

impl ChunkMetadata {
    /// Create a new `ChunkMetadata` instance with 64-byte SIMD width.
    pub fn new(chunk_id: u64, byte_offset: u64, byte_len: u64) -> Self {
        Self {
            chunk_id,
            byte_offset,
            byte_len,
            simd_width: 64,
        }
    }
}

/// Align a tensor payload to a 64-byte boundary for AVX-512-friendly processing.
pub fn align_tensor_avx512(data: &[u8]) -> AlignedVec {
    AlignedVec::new(data)
}
