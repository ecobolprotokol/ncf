use crate::schema::DType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::convert::TryInto;

/// Quantization levels used by the adaptive quantizer.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantLevel {
    /// 8-bit quantization.
    Q8,
    /// 4-bit quantization.
    Q4,
    /// 2-bit quantization.
    Q2,
    /// Lower-precision 4-bit K/M quantization.
    Q4_K_M,
}

impl QuantLevel {
    /// Number of bits used to represent each quantized element.
    pub fn bit_depth(self) -> usize {
        match self {
            QuantLevel::Q8 => 8,
            QuantLevel::Q4 | QuantLevel::Q4_K_M => 4,
            QuantLevel::Q2 => 2,
        }
    }
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
    pub fn quantize_tensor(
        name: &str,
        dtype: DType,
        payload: &[u8],
    ) -> (DType, Vec<u8>, QuantLevel) {
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
    pub fn build_quant_map(
        tensor_names: impl Iterator<Item = String>,
    ) -> BTreeMap<String, QuantLevel> {
        tensor_names
            .map(|name| (name.clone(), Self::assign_quant_level(&name)))
            .collect()
    }
}

fn quantize_f32(input: &[u8], bit_depth: usize) -> Vec<u8> {
    if input.len() % 4 != 0 || bit_depth == 0 {
        return input.to_vec();
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        min = min.min(value);
        max = max.max(value);
    }

    if !(min < max) {
        return vec![0u8; (input.len() / 4) * (bit_depth / 8).max(1)];
    }

    let levels = 1u32 << bit_depth;
    let range = max - min;
    let mut quantized = Vec::with_capacity((input.len() / 4) * ((bit_depth + 7) / 8));

    let mut raw_values = Vec::new();
    raw_values.reserve(input.len() / 4);
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        let normalized = ((value - min) / range).clamp(0.0, 1.0);
        let quant = (normalized * ((levels - 1) as f32)).round() as u32;
        raw_values.push(quant);
    }

    match bit_depth {
        8 => pack_q8(&raw_values, &mut quantized),
        4 => pack_q4_bitplanes(&raw_values, &mut quantized),
        2 => pack_q2_bitplanes(&raw_values, &mut quantized),
        _ => pack_bits(&raw_values, bit_depth, &mut quantized),
    }

    quantized
}

fn pack_q8(values: &[u32], out: &mut Vec<u8>) {
    out.reserve(values.len());
    for &value in values {
        out.push(value as u8);
    }
}

fn pack_q4_bitplanes(values: &[u32], out: &mut Vec<u8>) {
    for chunk in values.chunks(16) {
        let mut block = [0u16; 4];
        for (i, &value) in chunk.iter().enumerate() {
            let value = value as u16 & 0xF;
            for bit in 0..4 {
                block[bit] |= ((value >> bit) & 1) << i;
            }
        }
        for word in &block {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}

fn pack_q2_bitplanes(values: &[u32], out: &mut Vec<u8>) {
    for chunk in values.chunks(32) {
        let mut block = [0u32; 2];
        for (i, &value) in chunk.iter().enumerate() {
            let value = value as u32 & 0x3;
            for bit in 0..2 {
                block[bit] |= ((value >> bit) & 1) << i;
            }
        }
        for word in &block {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}

fn pack_bits(values: &[u32], bit_depth: usize, out: &mut Vec<u8>) {
    let mut bit_buffer = 0u64;
    let mut bit_count = 0;
    for &value in values {
        bit_buffer |= (value as u64) << bit_count;
        bit_count += bit_depth as u64;
        while bit_count >= 8 {
            out.push(bit_buffer as u8);
            bit_buffer >>= 8;
            bit_count -= 8;
        }
    }
    if bit_count > 0 {
        out.push(bit_buffer as u8);
    }
}

/// Decode packed quantized values back to integer representations with a static enum dispatch.
pub fn unpack_quantized_payload(
    dtype: DType,
    payload: &[u8],
    element_count: usize,
) -> Result<Vec<u32>, String> {
    match dtype {
        DType::Q8_0 => unpack_q8(payload, element_count),
        DType::Q4_0 | DType::Q4K => unpack_q4_bitplanes(payload, element_count),
        DType::Custom(2) => unpack_q2_bitplanes(payload, element_count),
        other => Err(format!(
            "unsupported quantized dtype for unpack: {:?}",
            other
        )),
    }
}

fn unpack_q8(payload: &[u8], element_count: usize) -> Result<Vec<u32>, String> {
    if payload.len() < element_count {
        return Err("q8 payload too short".to_string());
    }
    Ok(payload
        .iter()
        .take(element_count)
        .map(|&b| b as u32)
        .collect())
}

fn unpack_q4_bitplanes(payload: &[u8], element_count: usize) -> Result<Vec<u32>, String> {
    let expected = ((element_count + 15) / 16) * 8;
    if payload.len() < expected {
        return Err("q4 payload too short".to_string());
    }

    let mut output = Vec::with_capacity(element_count);
    let mut cursor = 0;
    for _ in 0..((element_count + 15) / 16) {
        let mut planes = [0u16; 4];
        for plane in planes.iter_mut() {
            *plane = u16::from_le_bytes([payload[cursor], payload[cursor + 1]]);
            cursor += 2;
        }
        for bit_index in 0..16 {
            let mut value = 0u32;
            for bit in 0..4 {
                value |= (((planes[bit] >> bit_index) & 1) as u32) << bit;
            }
            output.push(value);
        }
    }
    output.truncate(element_count);
    Ok(output)
}

fn unpack_q2_bitplanes(payload: &[u8], element_count: usize) -> Result<Vec<u32>, String> {
    let expected = ((element_count + 31) / 32) * 8;
    if payload.len() < expected {
        return Err("q2 payload too short".to_string());
    }

    let mut output = Vec::with_capacity(element_count);
    let mut cursor = 0;
    for _ in 0..((element_count + 31) / 32) {
        let mut planes = [0u32; 2];
        for plane in planes.iter_mut() {
            *plane = u32::from_le_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
            ]);
            cursor += 4;
        }
        for bit_index in 0..32 {
            let mut value = 0u32;
            value |= ((planes[0] >> bit_index) & 1) << 0;
            value |= ((planes[1] >> bit_index) & 1) << 1;
            output.push(value);
        }
    }
    output.truncate(element_count);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::DType;

    #[test]
    fn test_q4_bitplane_roundtrip() {
        let count = 37;
        let mut input = Vec::with_capacity(count);
        for i in 0..count {
            input.push((i % 16) as u32);
        }
        let mut out = Vec::new();
        pack_q4_bitplanes(&input, &mut out);
        let decoded = unpack_q4_bitplanes(&out, count).expect("unpack q4");
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_q2_bitplane_roundtrip() {
        let count = 70;
        let mut input = Vec::with_capacity(count);
        for i in 0..count {
            input.push((i % 4) as u32);
        }
        let mut out = Vec::new();
        pack_q2_bitplanes(&input, &mut out);
        let decoded = unpack_q2_bitplanes(&out, count).expect("unpack q2");
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_q8_roundtrip() {
        let count = 100;
        let input: Vec<u32> = (0..count).map(|i| (i % 256) as u32).collect();
        let mut out = Vec::new();
        pack_q8(&input, &mut out);
        let decoded = unpack_q8(&out, count).expect("unpack q8");
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_quantize_tensor_static_match() {
        let payload = (0..1024)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect::<Vec<u8>>();
        let (_dtype, quantized, level) =
            AdaptiveQuantizer::quantize_tensor("ffn_down", DType::F32, &payload);
        assert_eq!(level, QuantLevel::Q2);
        assert!(quantized.len() > 0);
    }
}
