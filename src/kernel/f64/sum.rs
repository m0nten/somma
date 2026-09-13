use crate::kernel::sum64;

#[inline(always)]
pub fn sum(x: &[f64]) -> f64 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { sum64::sum_avx2(x) }
    } else {
        sum64::sum(x)
    }
}

#[inline(always)]
pub fn par_sum(x: &[f64], chunk_size: usize) -> f64 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { sum64::par_sum_avx2(x, chunk_size) }
    } else {
        sum64::par_sum(x, chunk_size)
    }
}
