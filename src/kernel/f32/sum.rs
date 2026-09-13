use crate::kernel::sum32;

#[inline(always)]
pub fn sum(x: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { sum32::sum_avx2(x) }
    } else {
        sum32::sum(x)
    }
}

#[inline(always)]
pub fn par_sum(x: &[f32], chunk_size: usize) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { sum32::par_sum_avx2(x, chunk_size) }
    } else {
        sum32::par_sum(x, chunk_size)
    }
}
