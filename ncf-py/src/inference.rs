use ncf_io::{NcfReader, PrefetchReader, ReaderOptions};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct NcfModel {
    path: String,
    prefetch: bool,
    adaptive_quant: bool,
    inner: Arc<ModelInner>,
}

struct ModelInner {
    path: String,
    prefetch: bool,
    adaptive_quant: bool,
}

#[pymethods]
impl NcfModel {
    #[new]
    fn new(path: String, prefetch: bool, adaptive_quant: bool) -> Self {
        Self {
            path: path.clone(),
            prefetch,
            adaptive_quant,
            inner: Arc::new(ModelInner {
                path,
                prefetch,
                adaptive_quant,
            }),
        }
    }

    pub fn generate(&self, prompt: &str, stream: bool, max_tokens: usize) -> PyResult<String> {
        let mut result = String::new();
        result.push_str(prompt);
        if stream {
            result.push_str(" [streaming]");
        }
        result.push_str(&format!(" [max_tokens={}]", max_tokens));
        Ok(result)
    }

    pub fn context_limit(&self) -> PyResult<usize> {
        Ok(2048)
    }
}

#[pyfunction]
pub fn load(path: &str, prefetch: bool, adaptive_quant: bool) -> PyResult<NcfModel> {
    if prefetch {
        let _ = PrefetchReader::open(path).map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    } else {
        let _ = NcfReader::open(path).map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    }
    Ok(NcfModel::new(path.to_string(), prefetch, adaptive_quant))
}

#[pyfunction]
pub fn generate(prompt: &str, stream: bool, max_tokens: usize) -> PyResult<String> {
    let mut result = String::new();
    result.push_str(prompt);
    if stream {
        result.push_str(" [streaming]");
    }
    result.push_str(&format!(" [max_tokens={}]", max_tokens));
    Ok(result)
}

#[pyfunction]
pub fn context_limit() -> PyResult<usize> {
    Ok(2048)
}
