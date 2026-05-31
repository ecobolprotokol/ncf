use crate::schema::DType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Quantization levels used by the adaptive quantizer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantLevel {
    Q8,
    Q4,
    Q2,
    Q4_K_M,
}

/// Adaptive quantizer helper for tensor type hints.
pub struct AdaptiveQuantizer;

impl AdaptiveQuantizer {
    /// Assign a quantization level based on a tensor's name.
    pub fn assign_quant_level(name: &str) -> QuantLevel {
        let name = name.to_ascii_lowercase();
        if name.contains("attention_weight") {
            QuantLevel::Q8
        } else if name.contains("ffn_gate") || name.contains("ffn_down") {
            QuantLevel::Q2
        } else if name.contains("embedding") {
            QuantLevel::Q4
        } else {
            QuantLevel::Q4_K_M
        }
    }

    /// Quantize a payload for a given tensor name and input data type.
    pub fn quantize_tensor(name: &str, dtype: DType, payload: &[u8]) -> (DType, Vec<u8>, QuantLevel) {
        let level = Self::assign_quant_level(name);
        let quantized = match (level, dtype) {
            (QuantLevel::Q8, DType::F32) => quantize_f32(payload, 8),
            (QuantLevel::Q4, DType::F32) => quantize_f32(payload, 4),
            (QuantLevel::Q2, DType::F32) => quantize_f32(payload, 2),
            (QuantLevel::Q4_K_M, DType::F32) => quantize_f32(payload, 4),
            _ => payload.to_vec(),
        };

        let output_dtype = match level {
            QuantLevel::Q8 => DType::Q8_0,
            QuantLevel::Q4 => DType::Q4_0,
            QuantLevel::Q2 => DType::Custom(2),
            QuantLevel::Q4_K_M => DType::Q4K,
        };

        (output_dtype, quantized, level)
    }

    /// Build a lookup map for tensor quantization levels.
    pub fn build_quant_map(tensor_names: impl Iterator<Item = String>) -> BTreeMap<String, QuantLevel> {
        tensor_names
            .map(|name| (name.clone(), Self::assign_quant_level(&name)))
            .collect()
    }
}

fn quantize_f32(input: &[u8], bit_depth: usize) -> Vec<u8> {
    if input.len() % 4 != 0 || bit_depth == 0 {
        return input.to_vec();
    }

    let mut values = Vec::with_capacity(input.len() / 4);
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        min = min.min(value);
        max = max.max(value);
        values.push(value);
    }

    if !(min < max) {
        return vec![0u8; values.len()];
    }

    let levels = 1u32 << bit_depth;
    let range = max - min;
    let mut out = Vec::with_capacity((values.len() * bit_depth + 7) / 8);
    let mut bit_buffer = 0u32;
    let mut bit_count = 0;

    for value in values {
        let normalized = ((value - min) / range).clamp(0.0, 1.0);
        let quant = (normalized * ((levels - 1) as f32)).round() as u32;
        bit_buffer |= quant << bit_count;
        bit_count += bit_depth as u32;

        while bit_count >= 8 {
            out.push((bit_buffer & 0xFF) as u8);
            bit_buffer >>= 8;
            bit_count -= 8;
        }
    }

    if bit_count > 0 {
        out.push(bit_buffer as u8);
    }

    out
}
