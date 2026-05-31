//! Utilities for reading and writing NCF files (I/O helpers).
#![deny(missing_docs)]

/// Memory-mapped reader implementation.
pub mod mmap;
/// Borrowed reader API for safe zero-copy access.
pub mod reader;
/// Prefetching reader that hints the kernel and warms the next tensor.
pub mod prefetch;
/// KV delta replay streaming support.
pub mod kv_stream;
/// File writer utilities to create NCF files.
pub mod writer;
/// Streaming API placeholder (hidden until implemented).
pub mod stream;

pub use mmap::NcfMmap;
pub use prefetch::PrefetchReader;
pub use kv_stream::KvStreamReader;
pub use reader::{NcfReader, NcfReaderHandle, ReaderOptions};
pub use writer::NcfWriter;
#[doc(hidden)]
pub use stream::NcfStream;

#[cfg(test)]
mod roundtrip_tests;
