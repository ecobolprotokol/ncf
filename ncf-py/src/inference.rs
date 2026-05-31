use ncf_io::{NcfReader, PrefetchReader};
use pyo3::prelude::*;

#[pyclass]
pub struct NcfModel {
    path: String,
    prefetch: bool,
    adaptive_quant: bool,
}

#[pymethods]
impl NcfModel {
    #[new]
    fn new(path: String, prefetch: bool, adaptive_quant: bool) -> Self {
        Self {
            path,
            prefetch,
            adaptive_quant,
        }
    }

    pub fn generate(&self, prompt: &str, stream: bool, max_tokens: usize) -> PyResult<String> {
        let mut result = String::new();
        result.push_str(prompt);
        if stream {
            result.push_str(" [streaming]");
        }
        if self.prefetch {
            result.push_str(" [prefetch]");
        }
        if self.adaptive_quant {
            result.push_str(" [adaptive_quant]");
        }
        result.push_str(&format!(" [model={}]", self.path));
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
        let _ = PrefetchReader::open(path)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    } else {
        let _ = NcfReader::open(path)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
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
