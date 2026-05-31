use std::path::PathBuf;

#[test]
fn convert_functions_return_error_on_missing_input() {
    // Call conversion functions with a non-existent path and ensure they return an error
    let missing = PathBuf::from("/nonexistent_file_for_test.bin");
    let out = PathBuf::from("/tmp/should_not_be_created.ncf");
    // We only check that functions return Err for missing input; not executing heavy parsing.
    let res1 = ncf_convert::safetensors_to_ncf(&missing, &out, None, None);
    assert!(res1.is_err());
    let res2 = ncf_convert::gguf_to_ncf(&missing, &out, None, None);
    assert!(res2.is_err());
}
