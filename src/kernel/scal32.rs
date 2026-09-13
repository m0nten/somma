use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use std::arch::x86_64::*;

#[inline(always)]
pub fn par_scal(x: &mut [f32], a: f32, chunk_size: usize) {
    x.par_chunks_mut(chunk_size).for_each(|x| scal(x, a));
}

/// # Safety
/// This functon have custom SIMD construction.
/// Not recommended.
#[target_feature(enable = "avx2")]
pub unsafe fn _par_scal_avx2(x: &mut [f32], a: f32, chunk_size: usize) {
    x.par_chunks_mut(chunk_size).for_each(|x| unsafe { _scal_avx2(x, a) });
}

/// # Safety
/// This functon have custom SIMD construction.
/// Not recommended.
#[target_feature(enable = "avx2")]
pub unsafe fn _scal_avx2(x: &mut [f32], a: f32) {
    let len = x.len();
    let ptr_x = x.as_mut_ptr();
    let var_a = _mm256_set1_ps(a);

    let mut i = 0;
    while i + 32 <= len {
        let p0 = unsafe { ptr_x.add(i) };
        let p1 = unsafe { ptr_x.add(i + 8) };
        let p2 = unsafe { ptr_x.add(i + 16) };
        let p3 = unsafe { ptr_x.add(i + 24) };

        let v0 = unsafe { _mm256_loadu_ps(p0) };
        let v1 = unsafe { _mm256_loadu_ps(p1) };
        let v2 = unsafe { _mm256_loadu_ps(p2) };
        let v3 = unsafe { _mm256_loadu_ps(p3) };

        unsafe {
            _mm256_storeu_ps(p0, _mm256_mul_ps(v0, var_a));
            _mm256_storeu_ps(p1, _mm256_mul_ps(v1, var_a));
            _mm256_storeu_ps(p2, _mm256_mul_ps(v2, var_a));
            _mm256_storeu_ps(p3, _mm256_mul_ps(v3, var_a));
        }

        i += 32;
    }

    while i + 8 <= len {
        unsafe {
            let p0 = ptr_x.add(i);
            let v0 = _mm256_loadu_ps(p0);
            _mm256_storeu_ps(p0, _mm256_mul_ps(v0, var_a));
        };

        i += 8;
    }

    while i < len {
        x[i] *= a;
        i += 1;
    }
}

#[inline(always)]
pub fn scal(x: &mut [f32], a: f32) {
    for x in x.iter_mut() {
        *x *= a;
    }
}
