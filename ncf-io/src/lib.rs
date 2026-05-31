//! Utilities for reading and writing NCF files (I/O helpers).
#![deny(missing_docs)]

/// KV delta replay streaming support.
pub mod kv_stream;
/// Memory-mapped reader implementation.
pub mod mmap;
/// Prefetching reader that hints the kernel and warms the next tensor.
pub mod prefetch;
/// Borrowed reader API for safe zero-copy access.
pub mod reader;
/// Streaming API placeholder (hidden until implemented).
pub mod stream;
/// File writer utilities to create NCF files.
pub mod writer;

pub use kv_stream::KvStreamReader;
pub use mmap::NcfMmap;
pub use prefetch::PrefetchReader;
pub use reader::{NcfReader, NcfReaderHandle, ReaderOptions};
#[doc(hidden)]
pub use stream::NcfStream;
pub use writer::NcfWriter;

#[cfg(test)]
mod roundtrip_tests;
