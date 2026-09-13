use crate::{errors::SommaError, kernel::dot64};

#[inline(always)]
pub fn dot(x: &[f64], y: &[f64]) -> Result<f64, SommaError> {
    if x.len() != y.len() {
        return Err(SommaError::DifferentVectors);
    }

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Ok(unsafe { dot64::dot_avx2(x, y) })
    } else {
        Ok(dot64::dot_scalar(x, y))
    }
}

#[inline(always)]
pub fn par_dot(x: &[f64], y: &[f64], chunk_size: usize) -> Result<f64, SommaError> {
    if x.len() != y.len() {
        return Err(SommaError::DifferentVectors);
    }

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Ok(unsafe { dot64::par_dot_avx2(x, y, chunk_size) })
    } else {
        Ok(dot64::par_dot_scalar(x, y, chunk_size))
    }
}
