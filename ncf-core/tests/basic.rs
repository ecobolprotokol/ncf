use ncf_core::chunk::ChunkHeader;
use ncf_core::constants::CHUNK_HEADER_SIZE;
use ncf_core::index::{IndexEntry, NcfIndex};
use std::collections::BTreeMap;

#[test]
fn chunk_header_encode_decode_roundtrip() {
    let header = ChunkHeader {
        chunk_id: 42,
        flags: 1,
        uncompressed_len: 100,
        compressed_len: 100,
    };
    let bytes = header.encode();
    assert_eq!(bytes.len(), CHUNK_HEADER_SIZE as usize);
    let decoded = ChunkHeader::decode(&bytes).expect("decode");
    assert_eq!(decoded.chunk_id, header.chunk_id);
    assert_eq!(decoded.flags, header.flags);
    assert_eq!(decoded.uncompressed_len, header.uncompressed_len);
    assert_eq!(decoded.compressed_len, header.compressed_len);
}

#[test]
fn ncf_index_chunk_map_is_built_for_lookups() {
    let entries = vec![IndexEntry {
        chunk_id: 123,
        byte_offset: 1024,
        byte_len: 4096,
        tensor_name_hash: 0,
    }];
    let mut tensor_map = BTreeMap::new();
    tensor_map.insert("tensor".to_string(), 123);

    let index = NcfIndex::new(entries, tensor_map.clone());
    let entry = index.find_entry(123).expect("entry exists");
    assert_eq!(entry.byte_offset, 1024);

    let mut buffer = Vec::new();
    ciborium::ser::into_writer(&index, &mut buffer).expect("serialize index");
    let restored: NcfIndex = ciborium::de::from_reader(buffer.as_slice()).expect("deserialize index");
    let restored_entry = restored.find_entry(123).expect("restored entry exists");
    assert_eq!(restored_entry.byte_len, 4096);
    assert_eq!(restored.find_chunk_id("tensor"), Some(123));
}
