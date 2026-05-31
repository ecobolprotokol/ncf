use ncf_core::chunk::ChunkHeader;
use ncf_core::constants::CHUNK_HEADER_SIZE;

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
