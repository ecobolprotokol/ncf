use pyo3::prelude::*;

mod inference;
use inference::{context_limit, generate, load, NcfModel};

#[pymodule]
fn ncf_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NcfModel>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(context_limit, m)?)?;
    Ok(())
}
