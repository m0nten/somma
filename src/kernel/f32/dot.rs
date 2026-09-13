use crate::{errors::SommaError, kernel::dot32};

#[inline(always)]
pub fn dot(x: &[f32], y: &[f32]) -> Result<f32, SommaError> {
    if x.len() != y.len() {
        return Err(SommaError::DifferentVectors);
    }

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Ok(unsafe { dot32::dot_avx2(x, y) })
    } else {
        Ok(dot32::dot_scalar(x, y))
    }
}

#[inline(always)]
pub fn par_dot(x: &[f32], y: &[f32], chunk_size: usize) -> Result<f32, SommaError> {
    if x.len() != y.len() {
        return Err(SommaError::DifferentVectors);
    }

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Ok(unsafe { dot32::par_dot_avx2(x, y, chunk_size) })
    } else {
        Ok(dot32::par_dot_scalar(x, y, chunk_size))
    }
}
